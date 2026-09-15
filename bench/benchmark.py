"""
Benchmark harness for the CUDA kernel exercises.

This is the Python-side scaffolding: it shells out to the compiled CUDA
binaries, times them, and compares against a NumPy CPU baseline so you have
a sanity-check reference point. It does NOT replace writing the kernels —
run `make` first and fill in the TODOs in kernels/*.cu, or every call here
will just print FAIL.

Usage:
    python bench/benchmark.py
"""

import subprocess
import time
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BIN_DIR = ROOT / "bin"

SIZES = [1 << 16, 1 << 20, 1 << 24]  # ~65K, ~1M, ~16M elements


def time_binary(binary: str, n: int) -> tuple[bool, float]:
    """Run a compiled kernel binary, return (passed, wall_seconds)."""
    exe = BIN_DIR / binary
    if not exe.exists():
        print(f"  [skip] {exe} not built yet — run `make {binary}` first.")
        return False, float("nan")

    start = time.perf_counter()
    result = subprocess.run([str(exe), str(n)], capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    passed = "PASS" in result.stdout
    if not passed:
        print(f"  stdout: {result.stdout.strip()}")
        print(f"  stderr: {result.stderr.strip()}")
    return passed, elapsed


def numpy_vector_add(n: int) -> float:
    a = np.ones(n, dtype=np.float32)
    b = np.full(n, 2.0, dtype=np.float32)
    start = time.perf_counter()
    _ = a + b
    return time.perf_counter() - start


def numpy_reduce(n: int) -> float:
    a = np.ones(n, dtype=np.float32)
    start = time.perf_counter()
    _ = a.sum()
    return time.perf_counter() - start


def run_suite(name: str, binary: str, numpy_fn):
    print(f"\n=== {name} ===")
    print(f"{'n':>12} | {'cuda (s)':>10} | {'numpy (s)':>10} | status")
    print("-" * 52)
    for n in SIZES:
        passed, cuda_t = time_binary(binary, n)
        np_t = numpy_fn(n)
        status = "PASS" if passed else "FAIL/SKIP"
        cuda_str = f"{cuda_t:.6f}" if not np.isnan(cuda_t) else "n/a"
        print(f"{n:>12} | {cuda_str:>10} | {np_t:.6f} | {status}")


if __name__ == "__main__":
    print("Note: timings include process startup + host<->device transfer,")
    print("so this is a correctness/smoke-test harness, not a rigorous")
    print("microbenchmark. For real kernel timing, add cudaEvent-based")
    print("timers inside the .cu files once you're past the TODOs.\n")

    run_suite("vector_add", "vector_add", numpy_vector_add)
    run_suite("reduction", "reduction", numpy_reduce)
