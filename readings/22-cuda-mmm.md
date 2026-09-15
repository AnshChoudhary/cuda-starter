# Reading: How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance

**Source:** [Simon Boehm — CUDA MMM worklog (Dec 2022)](https://siboehm.com/articles/22/CUDA-MMM)  
**Role in this repo:** Phase 2 reading after `01_vector_add` and `02_reduction` pass and you understand *why* each TODO works.

Do the exercises yourself. This note is a map of the article, not a substitute for reading it or implementing the kernels.

## What it is

A worklog that starts from a naive CUDA SGEMM (`C = αAB + βC`, FP32) and iteratively optimizes until ~95% of cuBLAS FP32 on an A6000/A100. Goal is understanding GPU performance (coalescing, shared memory, occupancy, tiling), not shipping a cuBLAS replacement. Tensor cores / TF32 / BF16 are out of scope.

Companion code is linked from the article (and a benchmarking setup from wangzyon’s repo).

## Why it matters after Phase 1

| Phase 1 skill | Where it shows up in matmul |
| --- | --- |
| Thread/block indexing | Assigning work to `C` tiles |
| Host ↔ device memory | Still present, but less of the story |
| Shared memory + `__syncthreads__` | Cache-blocking tiles of `A`/`B` |
| Thinking about races / sync | Every shared-memory stage |

Matmul is where those primitives compound into arithmetic intensity and bandwidth.

## Kernel progression (article numbering)

Approximate relative performance vs cuBLAS FP32 on the author’s A6000 setup:

| Kernel | Idea | ~% of cuBLAS |
| --- | --- | --- |
| 1 Naive | One thread → one `C` element; uncoalesced GMEM | ~1% |
| 2 GMEM coalescing | Remap so warps hit consecutive addresses | ~9% |
| 3 SMEM cache-blocking | Tile `A`/`B` into shared memory along `K` | ~13% |
| 4 1D blocktiling | Multiple `C` results per thread | ~37% |
| 5 2D blocktiling | `TM×TN` results per thread; register caches | ~69% |
| 6 Vectorized access | Transpose `As`; `float4` / 128-bit loads | ~78% |
| 9 Autotuning | Sweep `BM/BN/BK/TM/TN` | ~85% |
| 10 Warptiling | Explicit warp hierarchy between block & thread | ~94% |
| 0 cuBLAS | Reference | 100% |

(Kernels 7–8 in the author’s numbering are bank-conflict experiments he dropped as net-slower.)

## Concepts to internalize (in order)

1. **Warps & coalescing** — 32 threads; consecutive `threadIdx` → consecutive addresses → fewer GMEM transactions.
2. **Shared memory tiling** — load a chunk, sync, compute, sync; raise reuse before touching GMEM again.
3. **Occupancy vs. resources** — registers / SMEM / threads per block trade off how many blocks fit on an SM.
4. **Arithmetic intensity** — FLOPs per byte moved; computing more outputs per thread amortizes loads.
5. **Hierarchy** — block tile → warp tile → thread tile; match how the hardware schedules work.
6. **Autotuning** — tile sizes are GPU- and shape-dependent; one kernel is not enough for all sizes (cuBLAS dispatches many).

## Suggested Phase 2 workflow in this repo

1. Read the article through Kernel 3 with the diagrams open.
2. Add kernels under `kernels/` (e.g. `03_sgemm_naive.cu` …) following the same Makefile / `bin/` pattern as Phase 1.
3. Keep a host-side correctness check against a CPU or cuBLAS reference before chasing FLOPs.
4. Only then push through tiling / vectorization / warptiling — one optimization per commit if you can.

## Open questions the article leaves for later

- Double buffering (GMEM→SMEM and SMEM→registers)
- Shared-memory bank conflicts
- Tensor-core / WMMA paths
- Size-specialized kernels (cuBLAS’s many SGEMM variants)

## Attribution

Article © Simon Boehm. Link and study here; do not treat this file as a full reprint of the post.
