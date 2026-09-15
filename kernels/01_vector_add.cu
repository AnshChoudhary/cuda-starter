// 01_vector_add.cu
//
// GOAL: out[i] = a[i] + b[i] for N elements, one thread per element.
// This is the "hello world" of CUDA — the point is to nail the mental model:
//   grid -> blocks -> threads, and how a global thread index maps to data.
//
// Compile:  make vector_add
// Run:      ./bin/vector_add
//
// Fill in every TODO. Nothing here is a trick — if it doesn't compile,
// read the error, it's telling you something real about the memory model.

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                               \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

// TODO 1: Write the kernel.
// - Compute a global thread index from blockIdx, blockDim, threadIdx.
// - Guard against index >= n (grids are usually padded past the data size).
// - out[i] = a[i] + b[i]
__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    // your code here
}

void vector_add(const float* h_a, const float* h_b, float* h_out, int n) {
    float *d_a, *d_b, *d_out;
    size_t bytes = n * sizeof(float);

    // TODO 2: Allocate d_a, d_b, d_out on the device with cudaMalloc.
    // TODO 3: Copy h_a -> d_a and h_b -> d_b with cudaMemcpy (Host to Device).

    // TODO 4: Pick a block size (try 256) and compute how many blocks you
    // need to cover n elements (careful with integer division / remainders).
    int threads_per_block = 0;   // fill in
    int num_blocks = 0;          // fill in

    vector_add_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_out, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO 5: Copy d_out -> h_out (Device to Host).
    // TODO 6: cudaFree everything you allocated.
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : (1 << 20);  // default: ~1M elements

    float* h_a = (float*)malloc(n * sizeof(float));
    float* h_b = (float*)malloc(n * sizeof(float));
    float* h_out = (float*)malloc(n * sizeof(float));

    for (int i = 0; i < n; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    vector_add(h_a, h_b, h_out, n);

    // Correctness check
    bool ok = true;
    for (int i = 0; i < n; i++) {
        if (h_out[i] != 3.0f) {
            ok = false;
            printf("Mismatch at %d: got %f, expected 3.0\n", i, h_out[i]);
            break;
        }
    }
    printf(ok ? "PASS (n=%d)\n" : "FAIL\n", n);

    free(h_a); free(h_b); free(h_out);
    return ok ? 0 : 1;
}
