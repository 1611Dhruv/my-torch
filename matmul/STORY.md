# Writing a matmul that reaches ~80% of cuBLAS

This is a writeup of optimizing a single-precision GPU matrix multiply from a
naive kernel to one that sustains roughly 80% of cuBLAS on an RTX 5090. The
method was the same at every step: profile with Nsight Compute, find the dominant
bottleneck, address it, verify correctness against a reference, and repeat.

The 5090 has a theoretical FP32 peak near 105 TFLOP/s. cuBLAS sustains about
75–80 on large matmuls. The goal was never to beat it, only to close the gap
enough to understand where that performance comes from.

## Results

Every kernel, measured at the same sizes against cuBLAS on this card:

```
                       TFLOP/s @ size
kernel                 4096   8192  16384
0  naive                7.5    6.5    6.5
1  shared tiling       10.0    9.1    9.0
2  register tiling     34.2   38.0   37.6
3  + float4            35.2   40.5   40.5
4  + warp tiling       39.0   44.2   44.8
4b + transpose As      40.7   44.0   46.2
5  + double buffering  58.2   62.4   55.0
   cuBLAS              71.8   78.3   74.2
```

The hand-written kernel peaks near 62 TFLOP/s (~80% of cuBLAS), and every stage
matches cuBLAS numerically to ~1e-6. The 16384 column reflects thermal throttling
after several back-to-back sweeps — cuBLAS drops with it, so the ratio is stable.

## Naive and shared memory

The naive kernel assigns one output element per thread, each reading a full row
of A and column of B from global memory. At ~7 TFLOP/s it is bandwidth-bound:
every thread computing a result in row `i` re-reads that entire row from the
slowest memory on the device.

Shared-memory tiling is the standard fix — load a tile once per block and serve
the threads that need it from on-chip memory. The gain here was smaller than I
expected (~10 TFLOP/s), because the L2 cache was already absorbing much of the
naive version's redundant traffic. The benefit grows at larger problem sizes,
where the working set stops fitting in L2.

## Register tiling

This was the first large jump. Instead of one output per thread, each thread owns
an 8×8 block of the output with an `acc[8][8]` accumulator held in registers. The
inner loop becomes an outer product: per slice of k, load an 8-element strip of A
and an 8-element strip of B from shared memory once, into registers, and issue 64
multiply-adds from those — sixteen shared-memory reads for sixty-four MACs.

That took it to ~34 TFLOP/s. The mental model that made the rest of the project
coherent: shared memory captures reuse *between* threads, registers capture reuse
*within* a thread, and every optimization from here is some version of moving a
value further up the memory hierarchy and extracting more work from it before it
falls back down.

## float4

The profiler now reported the kernel as L1-bound, with two specific issues:
uncoalesced output stores (using 4 of every 32 bytes per transaction) and
shared-memory bank conflicts. Both are addressed by moving 128 bits per
instruction instead of 32 — `float4` loads and stores.

The subtlety that cost some time: `float4` requires 16-byte-aligned addresses. I
initially treated that as an edge-of-the-matrix concern, but if the row stride
isn't a multiple of 4, every row after the first begins at a misaligned address —
the misalignment is spread through the interior of the matrix, not the boundary.
Restricting test sizes to clean multiples of 4 resolved it.

## Warp tiling

A warp is 32 threads executing the same instruction in lockstep. When several of
them request the same shared-memory address in one instruction, the hardware
broadcasts it to all of them at once, effectively for free. Warp tiling assigns
each thread its outputs deliberately, so that an entire row of threads needs the
same A values and an entire column needs the same B values. Scattered reads
collapse into broadcasts, and the bank conflicts disappear with them.

My first warp-tiled kernel was slower than the version before it, for two
reasons. I had reduced the thread count, which doubled the work and register
footprint per thread and collapsed occupancy; and I had left the A-side reads
scalar and strided, producing an 8-way bank conflict — worse than where I
started. Restoring the thread count and storing A transposed in shared memory (so
its reads are contiguous, like B's) fixed both. Applying the optimization and
making it faster turned out to be two separate problems.

## Double buffering

The two `__syncthreads()` in the main loop serialize memory and compute: load a
tile, compute on it, load the next, compute. The memory system is idle during the
math and vice versa. Double buffering keeps two shared-memory buffers and begins
loading the next k-slab while computing on the current one, hiding load latency
behind the math. This was the largest single improvement after register
tiling — roughly 46 to 62 TFLOP/s — and also the easiest place to introduce a
kernel that runs correctly on most inputs but returns wrong results at slab
boundaries.

## Two things worth recording

For a while the benchmark reported ~90 TFLOP/s and I believed it. On reading the
driver again I found it was timing `cublasSgemm`, which I had swapped in earlier
to measure the ceiling and never reverted. The actual kernel was ~62. The lesson
is less about the mistake than about how I caught it: only by building a real
side-by-side comparison instead of trusting a single impressive number.

The correctness check mattered more than its line count suggests. Coalescing and
double-buffering bugs do not crash — they produce fast, wrong answers, which is
the worst failure mode to have. A float64 reference from PyTorch and a direct diff
against cuBLAS caught boundary bugs the benchmark alone would have missed. On a
future kernel I would write the oracle before the optimizations.

The project reduces to one loop: profile, fix the dominant bottleneck, confirm
correctness, repeat. Twice the profiler directed me somewhere I would not have
chosen — fixing the output stores before the bank conflicts, which felt backwards
and was not. That loop is the transferable part; the kernels are what it produced.
