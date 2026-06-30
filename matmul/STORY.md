# A matmul story: from 8 to ~63 TFLOP/s (and ~80% of cuBLAS)

This is the log of taking a single-precision matrix multiply on an **RTX 5090
(Blackwell, ~105 TFLOP/s FP32 peak)** from a naive kernel to a hand-written
tiled SGEMM that lands within ~80% of cuBLAS — and, just as importantly, *how*
each step was chosen: by profiling, finding the one bottleneck, and attacking it.

The rule the whole way: **let the profiler pick the next move.** Every jump
below came from reading an Nsight Compute report, finding the tallest
Speed-of-Light bar, and fixing exactly that.

---

## The ladder

Every kernel, measured at the **same `4096³`** size against cuBLAS on this card
(`make run`, Section 1). All seven agree with cuBLAS to ~`5e-6` — the ladder is a
performance story, not a correctness one; correctness held the whole way.

| Stage | What changed | Bottleneck it attacked | TFLOP/s | % cuBLAS |
|---|---|---|---|---|
| 0. Naive | one thread per output, read straight from global | global bandwidth | 7.5 | 10% |
| 1. Shared-memory tiling | block-cooperative `TILE×TILE` tiles in smem | global bandwidth → smem | 10.0 | 14% |
| 2. 2D register tiling | each thread owns an `8×8` register tile (outer product) | smem traffic / MIO | 34.1 | 48% |
| 3. `float4` vectorization | 128-bit loads/stores, coalesced + fewer instrs | L1 instruction pressure | 35.1 | 49% |
| 4. Warp tiling | lane layout → broadcast smem reads | L1 bank conflicts | 39.0 | 54% |
| 4b. Transpose `As` | contiguous `float4` A read (kills 8-way conflict) | the A-side bank conflict | 41.5 | 58% |
| 5. Double buffering | prefetch next k-slab while computing | load↔compute serialization | 58.2 | **81%** |
| — cuBLAS | the reference ceiling | — | 71.7 | 100% |

Two honest notes from the measured ladder: `float4` and warp tiling look modest
*at this size* (L2 hides some of what they fix), but pay off more at larger sizes
and under thermal load; and **double buffering was the biggest single jump after
register tiling** (+40%) — hiding load latency mattered a lot.

---

## 0 → 1: the memory hierarchy shows up

The naive kernel gives each thread one output and reads its whole row of `A` and
column of `B` from global memory. Every value of `A[i][:]` is re-read by every
thread in row `i` — enormous redundant traffic. It's bandwidth-bound and slow.

**Shared-memory tiling** fixes the *cross-thread* reuse: a block cooperatively
loads a `TILE×TILE` patch of `A` and `B` into shared memory once, and the
`TILE` threads that need each value read it from fast on-chip smem instead of
global. ~`TILE×` less global traffic. First big jump.

**Lesson:** shared memory is the only on-chip level the whole block can see, so
it's where *cross-thread* reuse has to live. Registers are private; they can't
do this job.

## 1 → 2: register tiling and the outer product

Shared tiling still reads from smem a lot. **2D register tiling** gives each
thread an `8×8` block of outputs and an `acc[8][8]` accumulator in registers.
The inner loop becomes an **outer product**: per `k`-slice, load an 8-vector
fragment of `A` and an 8-vector of `B` from smem *once* into registers, then do
64 multiply-adds entirely from registers.

```
16 smem reads  →  64 MACs   (per kk)   ← 8× arithmetic-intensity win
```

**Lesson:** the two tiling levels capture two different axes of reuse —
shared = cross-thread, registers = within-thread. You need both.

## 2 → 3: `float4`

The profiler now said **L1-bound**, with uncoalesced global stores (the
writeback used only 4 of every 32 bytes per sector — `Est. Speedup: 67%`) and
shared bank conflicts (`3.2-way`, `Est. Speedup: 38%`). Both are fixed by
**128-bit memory instructions**: `float4` loads/stores move 4 contiguous floats
per instruction → 4× fewer instructions, full sectors, broadcast-friendly smem
reads.

The trap: `float4` needs **16-byte-aligned** addresses, which means the row
stride (`k` for `A`, `m` for `B`) must be a multiple of 4 — not just the column
offset. Misalignment hits *most rows* when the stride isn't a multiple of 4, not
just the tail. So: test on multiple-of-4 (and multiple-of-128) sizes.

**Lesson:** correctness tests can't catch a coalescing bug — it's *correct*,
just slow. Only the profiler (sectors/request, bank conflicts) shows it.

## 3 → 4: warp tiling, or "seat the threads so they share cards"

Still L1-bound (`~81%`), now from smem-read *volume*: each value was loaded ~16×
because every thread fetched its fragment independently.

**Warp tiling** reshapes each warp's output footprint from a wide `2×16` strip
into a square-ish block, and lays out its 32 lanes as a grid so that:

- lanes in the same **lane-row** want the same `A` values → one **broadcast**,
- lanes in the same **lane-column** want the same `B` values → one broadcast.

A broadcast serves many lanes from one smem read (and isn't a bank conflict), so
the redundant smem traffic collapses and L1 drops.

This is the stage that bit back. The first attempt was **slower** — two
self-inflicted wounds:

1. **Halved the thread count** (128 instead of 256) → each thread computed 128
   outputs → `acc[8][16]` = ~150 registers → occupancy crashed to ~19%.
2. **Left the `A` read scalar and strided** → an **8-way** bank conflict (worse
   than before).

Fixes: back to 256 threads / 64 outputs per thread, and **transpose `As` in
shared** so the `A` read is contiguous `float4` like `B`. That recovered and
then beat the previous best.

**Lesson:** warp tiling is the most parameter-sensitive step on the ladder. The
*idea* (broadcast layout) is easy to get right and the *tuning* (threads,
registers, both fragment reads conflict-free) is easy to get wrong.

## 4 → 5: double buffering

The two `__syncthreads()` in the loop serialize memory and compute: load, then
compute, then load. The global latency of the next slab is exposed, not hidden.

**Double buffering** uses two smem buffers and prefetches slab `p+1` while
computing on slab `p`, hiding load latency behind compute. Final stage, and the
most bug-prone — easy to write something fast that's subtly wrong at slab
boundaries, which is exactly why the oracle matters.

---

## Final numbers (`results/`, clean sweep)

```
  size     yours    cuBLAS   %ceil   correctness(vs cuBLAS)
   512       5.4      14.9    36.3%   maxrel=1.5e-06   PASS
  1024      22.9      45.4    50.5%   maxrel=2.2e-06   PASS
  2048      52.5      68.4    76.8%   maxrel=3.1e-06   PASS
  4096      58.5      71.9    81.4%   maxrel=5.0e-06   PASS
  8192      62.9      77.7    81.0%   maxrel=7.4e-06   PASS
 16384      56.2      74.3    75.7%   maxrel=0.0e+00   PASS
```

- **Correctness:** agrees with cuBLAS to ~`1e-6` (float32) and passes the float64
  PyTorch oracle (`max abs err 5.9e-4`). It's not just fast — it's right.
- **Peak ~63 TFLOP/s** (8192), **~80% of cuBLAS** in the sweet spot.
- Small sizes (512, 1024) underutilize the 170 SMs — too few blocks to fill the
  GPU. Universal; cuBLAS dips there too.
- These are **steady-state, thermally throttled** numbers (a back-to-back sweep
  heats the card and clocks drop). Cold single-shot runs were higher (~72
  ours / ~90 cuBLAS) — same ~80% ratio either way, which is the robust metric.

---

## What actually carried the project

- **Profile, don't guess.** Every step came from the tallest Speed-of-Light bar.
  Twice the profiler *redirected* the plan (float4 before warp tiling; the
  writeback before the bank conflicts).
- **A correctness oracle is non-negotiable.** Coalescing and double-buffering
  bugs don't show up as crashes — they show up as fast wrong answers. The
  PyTorch float64 oracle (and the direct cuBLAS comparison) caught what the
  benchmark couldn't.
- **The memory hierarchy is the whole game.** global → shared → registers, each
  capturing a different axis of reuse. Every optimization was really about
  moving a byte one rung up the hierarchy and reusing it more before it fell
  back down.
