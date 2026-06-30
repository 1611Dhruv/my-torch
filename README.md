# MyTorch

A from-scratch deep learning framework in C++ with a CUDA backend — a small
PyTorch, built to actually understand the internals: from the `Tensor` up through
autograd, nn layers, a Transformer, and the GPU kernels underneath.

The bar throughout: **correctness gets validated against PyTorch** (dump a
reference, `allclose` in float64), and **kernels get profiled** with Nsight
Compute against a roofline. The point isn't to beat PyTorch or cuBLAS — it's to
understand *why* every piece works the way it does.

---

## The build, bottom to top

* **Tensor** — flat storage plus shape, strides, and offset. Broadcasting via
  stride-0, zero-copy views (transpose, slice, reshape), and a contiguity check
  before anything hits a kernel.
* **Autograd** — reverse-mode over a dynamic compute graph. Each op records its
  parents and a backward closure; `.backward()` topo-sorts and accumulates grads.
* **nn layers** — composable modules (linear, activations, layernorm, embedding).
* **Transformer** — attention, MLP blocks, the usual stack.
* **Adam** — with bias correction.
* **CUDA kernels** — elementwise ops, then a tiled, shared-memory matmul closing
  in on cuBLAS throughput.
* **FlashAttention** — online softmax with SRAM tiling, so the O(n²) attention
  matrix never gets materialized in HBM.

---

## Repo layout — how to navigate

```
my-torch/
├── include/mytorch/        Public headers (the API surface)
│   ├── tensor.h            Tensor: shape/strides/offset, views, contiguity
│   ├── storage.h           Flat reference-counted buffer underneath a Tensor
│   ├── ops.h               Op declarations (elementwise, matmul, dispatch)
│   ├── autograd.h          Compute graph node + backward-closure machinery
│   ├── cuda_utils.h        CUDA error-checking / launch helpers
│   └── nn/
│       ├── module.h        Base Module (params, forward)
│       └── linear.h        Linear layer
│
├── src/                    Implementations (one .cpp/.cu per header concern)
│   ├── tensor.cpp          Tensor logic: views, broadcasting, indexing
│   ├── storage.cpp         Buffer allocation / lifetime
│   ├── ops/
│   │   ├── dispatch.cpp    Routes ops to the CPU or CUDA backend
│   │   ├── elementwise.cpp / .cu   Elementwise ops (host + device)
│   │   └── matmul.cpp / .cu         Matmul (host entry + device kernel)
│   └── autograd/
│       └── functional.cpp  Reverse-mode backward passes
│
├── tests/                  GoogleTest suites (tensor, storage, elementwise,
│                           matmul, autograd) — the correctness net
├── examples/
│   └── hello_world.cpp     Smallest "it links and runs" sanity check
│
├── matmul/                 Standalone SGEMM optimization study (see below)
│   ├── src/                kernels.cu, driver.cpp, matmul.h
│   ├── bench/oracle.py     float64 PyTorch oracle
│   ├── results/            saved benchmark + Nsight outputs
│   └── STORY.md            the full optimization journey
│
└── CMakeLists.txt          Core lib + tests + examples (CUDA sm_120 / Blackwell)
```

**Where to start reading:** `include/mytorch/tensor.h` → `src/tensor.cpp` is the
foundation everything else stands on. Then `ops/dispatch.cpp` to see how a call
reaches a kernel, and `autograd/functional.cpp` for how gradients flow back.

### Build & run

```sh
cmake -B build -S .       # configure (needs CUDA toolkit; pinned to sm_120)
cmake --build build       # build the mytorch lib + tests + examples
ctest --test-dir build    # run the correctness suites
```

---

## A matmul story: from ~8 to ~63 TFLOP/s (≈80% of cuBLAS)

The [`matmul/`](matmul/) directory is a self-contained study: taking a single-precision
matrix multiply on an **RTX 5090 (Blackwell, ~105 TFLOP/s FP32 peak)** from a
naive kernel to a hand-written tiled SGEMM within ~80% of cuBLAS — and matching
it numerically.

The rule the whole way: **let the profiler pick the next move.** Every jump below
came from reading an Nsight Compute report, finding the tallest Speed-of-Light
bar, and fixing exactly that.

### The ladder

Every kernel, measured at the same `4096³` size against cuBLAS. All seven agree
with cuBLAS to ~`5e-6` — this is a *performance* story, not a correctness one;
correctness held the whole way.

| Stage | What changed | Bottleneck it attacked | TFLOP/s | % cuBLAS |
|---|---|---|---|---|
| 0. Naive | one thread per output, straight from global | global bandwidth | 7.5 | 10% |
| 1. Shared-memory tiling | block-cooperative `TILE×TILE` tiles in smem | global → smem | 10.0 | 14% |
| 2. 2D register tiling | each thread owns an `8×8` register tile (outer product) | smem traffic / MIO | 34.1 | 48% |
| 3. `float4` vectorization | 128-bit loads/stores, coalesced + fewer instrs | L1 instruction pressure | 35.1 | 49% |
| 4. Warp tiling | lane layout → broadcast smem reads | L1 bank conflicts | 39.0 | 54% |
| 4b. Transpose `As` | contiguous `float4` A read (kills 8-way conflict) | the A-side bank conflict | 41.5 | 58% |
| 5. Double buffering | prefetch next k-slab while computing | load↔compute serialization | 58.2 | **81%** |
| — cuBLAS | the reference ceiling | — | 71.7 | 100% |

The two big levers: **register tiling** (the outer product turns 16 smem reads
into 64 MACs — an 8× arithmetic-intensity win) and **double buffering** (+40%,
the biggest single jump after register tiling — hiding load latency mattered a
lot).

### Final sweep (`matmul/results/`)

```
  size     yours    cuBLAS   %ceil   correctness(vs cuBLAS)
   512       5.4      14.9    36.3%   maxrel=1.5e-06   PASS
  1024      22.9      45.4    50.5%   maxrel=2.2e-06   PASS
  2048      52.5      68.4    76.8%   maxrel=3.1e-06   PASS
  4096      58.5      71.9    81.4%   maxrel=5.0e-06   PASS
  8192      62.9      77.7    81.0%   maxrel=7.4e-06   PASS
 16384      56.2      74.3    75.7%   maxrel=0.0e+00   PASS
```

Peak **~63 TFLOP/s** (steady-state, thermally throttled), **~80% of cuBLAS** in
the sweet spot, correct to ~`1e-6` against cuBLAS and the float64 PyTorch oracle.

**Three things that carried it:** profile, don't guess (twice the profiler
*redirected* the plan); a correctness oracle is non-negotiable (coalescing and
double-buffering bugs are fast *wrong* answers, not crashes); and the memory
hierarchy is the whole game (global → shared → registers, each a different axis
of reuse).

→ Full write-up with the per-stage reasoning: **[matmul/STORY.md](matmul/STORY.md)**

---

## Game plan

The road from a flat buffer to a Transformer that trains. Checked = landed.

**Tensor & storage**
- [x] Flat `Storage` buffer with reference counting
- [x] `Tensor`: shape, strides, offset
- [x] Zero-copy views — transpose, slice, reshape
- [x] Broadcasting via stride-0
- [x] Contiguity check before kernel dispatch
- [ ] `.to(device)` host↔device transfer ergonomics

**Ops & backend**
- [x] Elementwise ops (CPU + CUDA)
- [x] CPU/CUDA dispatch layer
- [x] Tiled shared-memory matmul kernel
- [x] SGEMM optimization ladder → ~80% of cuBLAS *(see the story above)*
- [ ] Reductions (sum, mean, max) with backward
- [ ] Softmax / layernorm kernels

**Autograd**
- [x] Dynamic compute graph (parents + backward closures)
- [x] Topo-sort `.backward()` with grad accumulation
- [ ] Backward coverage for every forward op
- [ ] Gradient check harness vs the PyTorch oracle

**nn & training**
- [x] `Module` base + `Linear`
- [ ] Activations (ReLU, GELU)
- [ ] LayerNorm, Embedding
- [ ] Adam with bias correction
- [ ] A toy model that trains end-to-end

**Transformer**
- [ ] Scaled dot-product attention
- [ ] Multi-head attention + MLP block
- [ ] Full Transformer stack
- [ ] **FlashAttention** — online softmax, SRAM tiling, no O(n²) matrix in HBM

**Distributed & scaling**
- [ ] Multi-GPU data parallel (NCCL all-reduce on grads)
- [ ] **FSDP** — fully sharded data parallel: shard params, grads, and optimizer
  state across GPUs; all-gather on demand for forward/backward, reduce-scatter grads

**Always-on**
- [x] GoogleTest suites as the correctness net
- [x] PyTorch float64 oracle for numerical truth
- [ ] Nsight Compute roofline pass on each new kernel
