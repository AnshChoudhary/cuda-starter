# CUDA Kernel Starter (Phase 1)

Skeleton exercises for learning real CUDA C++. Every kernel has TODOs —
filling them in is the point. Don't copy a solution from the internet before
you've genuinely tried; the value is in getting stuck on things like
off-by-one thread indexing and race conditions yourself.

## 1. Get a GPU

You need an NVIDIA GPU with the CUDA toolkit installed. Options, cheapest
first:

- **RunPod** (runpod.io) — on-demand pods, pay per hour, has a "Community
  Cloud" tier that's cheaper. Pick a template with CUDA preinstalled
  (e.g. "RunPod PyTorch").
- **Lambda Labs** (lambda.ai) — on-demand instances, simple pricing,
  reliably has CUDA + drivers set up.
- **vast.ai** — marketplace of GPU rentals, usually cheapest, more variance
  in reliability.

Any single GPU is fine for this phase (even an older one — you don't need
an H100 to learn indexing and shared memory). **Stop the instance when
you're not actively using it** — billing is per-second/minute on all three.

## 2. Verify your environment

SSH into the instance and check:

```bash
nvcc --version      # CUDA compiler
nvidia-smi          # confirms the GPU is visible
```

## 3. Build

```bash
make vector_add     # or `make` to build everything
```

If `-arch=native` in the Makefile fails on an older nvcc version, replace it
with your GPU's specific architecture flag, e.g. `-arch=sm_80` (A100),
`-arch=sm_90` (H100) — `nvidia-smi` or a quick search tells you which.

## 4. Run directly

```bash
./bin/vector_add          # default ~1M elements
./bin/vector_add 1000000  # or specify n
```

Should print `PASS` once your TODOs are correct. If it prints `FAIL` or
segfaults, that's normal — read the error, check your indexing, and the
`CUDA_CHECK` macro will point at the exact failing line.

## 5. Benchmark harness (Python)

```bash
pip install numpy
python bench/benchmark.py
```

This runs both kernels across a few sizes and prints wall-clock time next
to a NumPy CPU baseline, so you can see where the GPU starts winning (and
notice it *doesn't* win at tiny sizes — transfer overhead dominates, which
is itself a useful lesson).

## Exercise order

1. `01_vector_add.cu` — thread/block indexing, malloc/memcpy/free, the
   basic host<->device dance.
2. `02_reduction.cu` — shared memory, `__syncthreads()`, tree reduction.
   This is the one that actually teaches you something vector_add doesn't.

## What's next

Once both pass and you understand *why* each TODO works (not just that it
compiles), move to Simon Boehm's CUDA matmul worklog
([siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM))
and implement each of his kernels yourself in this same project structure —
that's Phase 2.

See [`readings/22-cuda-mmm.md`](readings/22-cuda-mmm.md) for a roadmap of
the article's progression and how it connects to Phase 1 concepts.

See [`readings/cuda-matmul-kernel-agents.md`](readings/cuda-matmul-kernel-agents.md)
for a primer on the CUDA-MMM article, who needs kernel agents, and sequencing
advice for targeting NVIDIA's SOL-ExecBench leaderboard.
