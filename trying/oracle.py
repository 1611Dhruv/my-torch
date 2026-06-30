"""
PyTorch oracle for the CUDA matmul.

Flow:
  1. Your C++ driver dumps A, B, and the kernel's output C to raw binary files.
  2. This script loads them, recomputes A @ B in float64 (the "truth"),
     and checks your float32 kernel against it with allclose.

File format (self-describing, so Python doesn't need to be told the shapes):
     [int32 rows][int32 cols][float32 * rows*cols]   all little-endian, row-major

Run:  python oracle.py            # expects A.bin, B.bin, C.bin in cwd
"""

import sys
import numpy as np
import torch


def load(path):
    """Read a [rows][cols][data...] blob written by the C++ dump() helper."""
    with open(path, "rb") as f:
        rows, cols = np.fromfile(f, dtype=np.int32, count=2)
        data = np.fromfile(f, dtype=np.float32, count=int(rows) * int(cols))
    return data.reshape(int(rows), int(cols))


def main():
    A = load(sys.argv[1] if len(sys.argv) > 1 else "A.bin")
    B = load(sys.argv[2] if len(sys.argv) > 2 else "B.bin")
    C = load(sys.argv[3] if len(sys.argv) > 3 else "C.bin")

    # shape sanity — catches a wrong n/k/m before the numbers ever do
    n, k = A.shape
    k2, m = B.shape
    assert k == k2, f"inner dims disagree: A is {A.shape}, B is {B.shape}"
    assert C.shape == (n, m), f"C should be {(n, m)}, got {C.shape}"

    # float64 reference = the oracle's "truth"
    ref = torch.from_numpy(A).double() @ torch.from_numpy(B).double()
    got = torch.from_numpy(C).double()

    # Your kernel accumulates in float32, so it won't match to the bit.
    # Tolerance has to allow float32 rounding accumulated over k terms.
    rtol, atol = 1e-3, 1e-3
    if torch.allclose(got, ref, rtol=rtol, atol=atol):
        print(f"PASS  ({n}x{k} @ {k}x{m})  max abs err = {(got - ref).abs().max():.2e}")
        return 0

    diff = (got - ref).abs()
    flat = int(diff.argmax())
    i, j = np.unravel_index(flat, diff.shape)
    print(f"FAIL  ({n}x{k} @ {k}x{m})")
    print(f"  max abs err : {diff.max():.4e}   (rtol={rtol}, atol={atol})")
    print(f"  worst cell  : C[{i},{j}]  got={got[i, j]:.6f}  ref={ref[i, j]:.6f}")
    print(f"  mismatched  : {(diff > atol + rtol * ref.abs()).sum().item()} / {n * m} cells")
    return 1


if __name__ == "__main__":
    sys.exit(main())
