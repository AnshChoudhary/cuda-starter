# CUDA Matmul & AI Kernel-Generation Agents — Summary

## 1. Primer: concepts for Simon Boehm's "How to Optimize a CUDA Matmul Kernel" article

**Article:** [siboehm.com/articles/22/CUDA-MMM](https://siboehm.com/articles/22/CUDA-MMM)

Local map: [readings/22-cuda-mmm.md](./22-cuda-mmm.md)

### The problem

SGEMM: \(C = \alpha\cdot A\cdot B + \beta\cdot C\) in single precision (fp32).

Multiplying two \(N\times N\) matrices costs \(\sim 2N^3\) FLOPs — each output entry is a dot product of length \(N\), and each step is a multiply + an add.

**FMA (fused multiply-add):** one hardware instruction computing \(a\cdot b + c\).

- **Speed:** matmul is almost entirely FMAs, which is why it can approach a GPU's advertised peak FLOP/s (that peak assumes FMA throughput).
- **Fusion:** the multiply's full-width intermediate result is kept internally and only rounded once, after the add — instead of rounding once after the multiply and again after the add. Fewer roundings over a long dot product (e.g. length 4092) means better accuracy.
- Still counts as **2 FLOPs** even though it's 1 instruction — this is where the \(2N^3\) comes from.

### GPU execution hierarchy (software side)

- Grid → blocks → threads (up to 1024 threads/block).
- You write kernel code from a single thread's perspective; `blockIdx` / `threadIdx` tell each thread which piece of work it owns.

### GPU execution hierarchy (hardware side)

- **SM (streaming multiprocessor):** physical core cluster; blocks are scheduled onto SMs.
- **Warp:** group of 32 threads with consecutive thread IDs, scheduled and executed together. Most optimizations in the article are really about being warp-friendly.

### Memory hierarchy (fast → slow)

| Tier | Scope | Approx. bandwidth | Notes |
| --- | --- | --- | --- |
| Registers | per-thread | fastest | |
| Shared memory (SMEM) | per-block, on-chip | ~12 TB/s | Manually managed; `__syncthreads()` coordinates access |
| Global memory (GMEM) | whole GPU | ~750 GB/s | What "GPU memory" specs refer to |

### The central idea: arithmetic intensity

**Arithmetic intensity** = FLOPs performed per byte moved.

Napkin math in the article: GPU can do far more FLOP/s than GB/s, so a good kernel should be **compute-bound**, not memory-bound.

- **Tiling / cache blocking:** load a chunk of data into fast memory, do all possible work on it before evicting it.
- **Block tiling:** tile into shared memory.
- **Thread tiling:** each thread computes a small square of outputs (not just one, not just a strip) to maximize reuse of loaded values.
- **Warp tiling:** an extra tiling level between block and thread tiling.
- **Roofline plot:** arithmetic intensity (x) vs. achievable FLOP/s (y) — a sloped memory-bound region and a flat compute-bound ceiling.

### Coalescing

If the 32 threads in a warp read consecutive, aligned addresses, the hardware merges them into one wide transaction (e.g. 128 bytes at once).

Scattered access → many small transactions → wasted bandwidth.

One of the cheapest wins in the article (kernel 2): a 2-line indexing change gets ~6× speedup purely from enabling coalescing.

### Occupancy & latency hiding

**Occupancy** = active warps per SM ÷ max possible active warps per SM.

Matters because GPUs hide instruction latency by having other warps ready to run. Capped by registers/thread, threads/block, and SMEM/block.

High occupancy is a means, not the goal — some kernels are fine at ~66%.

Warp stall reasons (from the profiler) diagnose why a kernel is slow, e.g. "Stall MIO Throttle" = warps waiting on shared-memory instructions.

### Other terms that appear later in the article

- **Vectorized access (`float4`):** one instruction moves 16 bytes instead of 4.
- **Bank conflicts:** shared memory is split into banks; same-bank access by threads in a warp serializes.
- **Double buffering:** prefetch next tile while computing on the current one.
- **Autotuning:** optimal tile sizes (`BM`, `BN`, `BK`, `TM`, `TN`) differ per GPU model — found via brute-force search.
- **cuBLAS:** NVIDIA's closed-source reference library; wins on small matrices partly because it's actually hundreds of kernels dispatched by shape at runtime, not one kernel.
- **PTX / SASS:** NVIDIA's virtual / actual assembly, used to verify what the compiler really generated.

### How to read the article

Go in order — each kernel exists because profiling revealed a specific flaw in the previous one. Skim the PTX/SASS/Godbolt dumps on a first pass; spend real time on the block-tiling and 2D-tiling diagrams, since they make the indexing math much easier to follow.

---

## 2. Who needs an AI agent that generates efficient GPU kernels?

Several real systems already exist doing this (NVIDIA's AVO, Cursor + NVIDIA's multi-agent system, Sakana's AI CUDA Engineer, CUDA Agent, etc.), some beating cuBLAS-class baselines. Groups who'd feel it most:

- **Frontier AI labs** training/serving large models — hot kernels (attention, matmul, MoE, normalization) dominate GPU-hour cost; small % gains compound at scale.
- **GPU hardware vendors (esp. NVIDIA)** — every new architecture needs its entire kernel library re-tuned from scratch, normally months of specialist engineering per launch.
- **Inference-serving companies / clouds** — kernel efficiency is close to their entire cost structure; agentic systems have already found wins specific to serving/decode workloads.
- **Builders on non-NVIDIA or emerging hardware** (AMD, accelerator startups) — they lack NVIDIA's decade of hand-tuning; an agent lets them shortcut that gap.
- **Teams with unusual/custom ops** (novel sparse attention, custom quantization) — no existing hand-tuned library covers them; agentic systems have shown 100×+ speedups here.
- **Smaller labs/startups without kernel specialists** — GPU performance engineers are rare and expensive; an agent is a way to buy that expertise on demand.

**Common thread:** not "who wants faster kernels" (everyone) but "who is currently bottlenecked on the small number of humans who can write them."

---

## 3. Should a new multi-agent kernel system target NVIDIA's SOL-ExecBench leaderboard?

**Leaderboard:** [research.nvidia.com/benchmarks/sol-execbench/leaderboard](https://research.nvidia.com/benchmarks/sol-execbench/leaderboard)

### What it is

- Real benchmark: 235 kernel problems drawn from 124 production AI models across six domains, forward + backward passes, BF16/FP8/NVFP4 precision, targeting the NVIDIA B200 GPU.
- Scored via **SOL-Score:** how much of the gap between a baseline runtime and the hardware Speed-of-Light bound (computed analytically via SOLAR) a submission closes — not just "faster than a software baseline."
- Multi-DSL: supports PyTorch, Triton, CUTLASS, cuDNN, CuTe DSL, cuTile, CUDA C++.
- Explicitly intended for "researchers and engineers developing AI-based kernel generation systems" — i.e. exactly this use case.

### Why it's a good target

- Harder, more honest framing than "beat some reference implementation" — same roofline philosophy as the CUDA-MMM article, formalized.
- Not locked to raw CUDA C++.

### Catches to know before committing

- **Hardware-locked:** official runs happen on B200s with clocks pinned at 1500 MHz. Without B200 access, local iteration can't match official numbers — B200s aren't commodity hardware, though cloud providers rent them.
- **SOL bound is analytic, not always achievable:** SOLAR uses tensor shapes only, not values — it can't capture value-dependent compression, constant propagation, sparsity effects, etc. A mediocre score doesn't necessarily mean the agent did something wrong.

### Recommended sequencing

1. **Don't start here.** Start on KernelBench or the CUDA-MMM repo itself (Boehm's kernels as a ladder) — both give correctness + speedup signal without needing exotic hardware.
2. Once the agent loop works (generate → compile → check correctness → benchmark → mutate → repeat), point it at SOL-ExecBench for a public, hardware-grounded credibility check — renting B200 time rather than assuming you need to own one.
3. Treat **correctness verification / anti-hacking checks** as core infrastructure from day one — every serious system in this space (AVO, CUDA Agent, AI CUDA Engineer) treats this as first-class, since agents left unchecked will find degenerate ways to "win" a benchmark.
