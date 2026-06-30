# matmul

A single-precision GPU matrix multiply, written from scratch and optimized to
within ~80% of cuBLAS on an RTX 5090. [STORY.md](STORY.md) describes how it got
there, stage by stage; this file covers building and running it.

## Results

Every kernel, measured at `4096³` against cuBLAS on this card:

```
0  naive                7.5 TFLOP/s
1  shared tiling       10.0
2  register tiling     34.2
3  + float4            35.2
4  + warp tiling       39.0
4b + transpose As      40.7
5  + double buffering  58.2   (~80% of cuBLAS; peaks ~62 at 8192³)
   cuBLAS              71.8
```

Every stage matches cuBLAS to ~1e-6 and passes a float64 PyTorch reference.

## Building and running

Requires the CUDA toolkit (`nvcc`, cuBLAS) and a CUDA GPU. The correctness check
additionally uses [`uv`](https://github.com/astral-sh/uv) to fetch numpy/torch.

```sh
make run      # full ladder at 4096/8192/16384, every kernel vs cuBLAS
make oracle   # build, dump a 1024^3 run, and verify it against PyTorch (float64)
make clean
```

`make run` times every kernel from naive to the final double-buffered version at
three sizes and diffs each against cuBLAS, reporting TFLOP/s, percentage of the
cuBLAS ceiling, and maximum relative error.

## Layout

```
src/
  kernels.cu   all kernels, the cuBLAS reference, and the host launchers
  driver.cpp   benchmark and correctness harness
  matmul.h     host-side API
bench/
  oracle.py    float64 PyTorch correctness check
results/       per-stage benchmark outputs and Nsight Compute reports
```

The fast kernels require sizes that are multiples of the block tile (128) and of
4 (for `float4` alignment); the benchmark uses such sizes. Profiling was done
with Nsight Compute, which needs elevated permissions to read GPU counters.
