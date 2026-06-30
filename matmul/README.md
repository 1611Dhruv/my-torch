# matmul

A from-scratch single-precision matrix multiply for the GPU, optimized
profiler-step-by-profiler-step from a naive kernel to a tiled SGEMM that lands
within **~80% of cuBLAS** on an RTX 5090 — and matches it numerically.

The point isn't to beat cuBLAS; it's to understand *why* each optimization
moves the needle. See **[STORY.md](STORY.md)** for the full journey
(naive → shared tiling → register tiling → `float4` → warp tiling → double
buffering), with the bottleneck each stage attacked and the lesson it taught.

## Result (RTX 5090, FP32 peak ~105 TFLOP/s)

```
  size     yours    cuBLAS   %ceil   correctness(vs cuBLAS)
  2048      52.5      68.4    76.8%   maxrel=3.1e-06   PASS
  4096      58.5      71.9    81.4%   maxrel=5.0e-06   PASS
  8192      62.9      77.7    81.0%   maxrel=7.4e-06   PASS
 16384      56.2      74.3    75.7%   maxrel=0.0e+00   PASS
```

Peak **~63 TFLOP/s** (steady-state), **~80% of cuBLAS** in the sweet spot, and
correct to ~`1e-6` against cuBLAS / `5.9e-4` against a float64 PyTorch oracle.
(Full table incl. small sizes in [STORY.md](STORY.md).)

## Layout

```
matmul/
  src/
    kernels.cu   all kernels: naive → shared → reg → float4 → warp → double-buf,
                 plus the cuBLAS reference and host entry points
    driver.cpp   benchmark + correctness harness (yours vs cuBLAS)
    matmul.h     host-side API
  bench/
    oracle.py    float64 PyTorch oracle (the definitive correctness check)
  results/       saved benchmark outputs (.result) and Nsight reports (.ncu-rep)
  Makefile
```

## Build & run

Requirements: CUDA toolkit (`nvcc`, `cublas`), a CUDA GPU, and — for the oracle
— [`uv`](https://github.com/astral-sh/uv) (pulls `numpy`/`torch` on demand).

```sh
make run      # build + run: (1) the full kernel ladder at 4096^3, then
              #              (2) the final kernel vs cuBLAS across sizes
make oracle   # build, dump a 1024^3 case, and float64-verify it against PyTorch
make clean
```

`make run` prints two tables. **Section 1** times *every* kernel
(naive → shared → register → float4 → warp → +transpose → +double-buf) at one
size and checks each against cuBLAS. **Section 2** sweeps the final kernel across
square sizes. Both report TFLOP/s, % of the cuBLAS ceiling, and max relative
error against cuBLAS.

## Notes

- The fast kernel (`kernel_warptiled_doublebuf`) needs sizes that are multiples
  of the block tile (`128`) and of 4 (for `float4` alignment). The sweep uses
  such sizes; ragged sizes need the edge-guard paths in the earlier kernels.
- Benchmark numbers are steady-state (a back-to-back sweep thermally throttles
  the card); cold single-shot runs are higher, but the **% of cuBLAS** is the
  throttle-independent metric.
- Profiling was done with Nsight Compute (`ncu`); GPU performance counters need
  elevated permissions (`sudo ncu …` or the `NVreg_RestrictProfilingToAdminUsers=0`
  module param).
