// 02_reduction.cu
//
// GOAL: sum all N elements of an array into a single float.
// This is the exercise that actually teaches you shared memory and
// __syncthreads() — vector_add never touches either.
//
// Approach: each block reduces its chunk into one partial sum using
// shared memory, then you sum the (few) partial sums on the host or
// with a second kernel launch. Don't try to do it all in one kernel
// with atomics yet — that's a later optimization, not a starting point.
//
// Compile:  make reduction
// Run:      ./bin/reduction

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

#define BLOCK_SIZE 256

// TODO 1: Each block computes the sum of its slice of `in` into
// partial_sums[blockIdx.x].
//
// Steps:
//   a) declare a __shared__ float buffer of size BLOCK_SIZE
//   b) each thread loads one element (or 0 if out of bounds) into shared mem
//   c) __syncthreads()
//   d) tree reduction: for stride = blockDim.x/2; stride > 0; stride /= 2
//        if threadIdx.x < stride: shared[tid] += shared[tid + stride]
//        __syncthreads() each iteration
//   e) thread 0 writes shared[0] to partial_sums[blockIdx.x]
__global__ void reduce_kernel(const float* in, float* partial_sums, int n) {
    __shared__ float shared[BLOCK_SIZE];
    // your code here
}

float reduce_sum(const float* h_in, int n) {
    int threads_per_block = BLOCK_SIZE;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    float *d_in, *d_partial;
    // TODO 2: cudaMalloc d_in (n floats) and d_partial (num_blocks floats).
    // TODO 3: cudaMemcpy h_in -> d_in.

    reduce_kernel<<<num_blocks, threads_per_block>>>(d_in, d_partial, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // TODO 4: Copy d_partial back to host (num_blocks floats) and sum them
    // on the CPU to get the final answer. (Yes, this last step is serial —
    // that's fine for now. Doing it fully on-device is a later exercise.)
    float* h_partial = (float*)malloc(num_blocks * sizeof(float));
    float total = 0.0f;

    // TODO 5: cudaFree d_in, d_partial. free(h_partial).
    return total;
}

int main(int argc, char** argv) {
    int n = argc > 1 ? atoi(argv[1]) : (1 << 20);

    float* h_in = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) h_in[i] = 1.0f;  // sum should equal n

    float result = reduce_sum(h_in, n);
    printf("Sum = %f (expected %d) -> %s\n", result, n,
           (result == (float)n) ? "PASS" : "FAIL");

    free(h_in);
    return 0;
}
