#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define N 1024
#define RUNS 10

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

// Naive: one thread per output element, global memory only
__global__ void matmul_naive(const float *A, const float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int k = 0; k < n; k++)
        sum += A[row * n + k] * B[k * n + col];
    C[row * n + col] = sum;
}

// Tiled: TxT tiles of A and B staged in shared memory (block size == tile size)
template <int T>
__global__ void matmul_tiled(const float *A, const float *B, float *C, int n) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    int row = blockIdx.y * T + threadIdx.y;
    int col = blockIdx.x * T + threadIdx.x;
    float sum = 0.0f;
    for (int t = 0; t < n / T; t++) {
        As[threadIdx.y][threadIdx.x] = A[row * n + t * T + threadIdx.x];
        Bs[threadIdx.y][threadIdx.x] = B[(t * T + threadIdx.y) * n + col];
        __syncthreads();
        for (int k = 0; k < T; k++)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    C[row * n + col] = sum;
}

static void launch(int tiled, int bs, const float *A, const float *B, float *C) {
    dim3 block(bs, bs), grid(N / bs, N / bs);
    if (!tiled)        matmul_naive<<<grid, block>>>(A, B, C, N);
    else if (bs == 4)  matmul_tiled<4><<<grid, block>>>(A, B, C, N);
    else if (bs == 8)  matmul_tiled<8><<<grid, block>>>(A, B, C, N);
    else if (bs == 16) matmul_tiled<16><<<grid, block>>>(A, B, C, N);
    else               matmul_tiled<32><<<grid, block>>>(A, B, C, N);
}

int main(void) {
    size_t bytes = (size_t)N * N * sizeof(float);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("GPU: %s (compute capability %d.%d)\n\n", prop.name, prop.major, prop.minor);

    float *hA = (float *)malloc(bytes), *hB = (float *)malloc(bytes);
    float *hC = (float *)malloc(bytes), *hRef = (float *)malloc(bytes);

    // Same inputs as the CPU threads: srand(10), A then B (scaled to [0,1])
    srand(10);
    for (int i = 0; i < N * N; i++) hA[i] = rand() / (float)RAND_MAX;
    for (int i = 0; i < N * N; i++) hB[i] = rand() / (float)RAND_MAX;

    // CPU reference (handout's matmul; i-k-j order for speed, same sum order per element)
    for (int i = 0; i < N * N; i++) hRef[i] = 0.0f;
    for (int i = 0; i < N; i++)
        for (int k = 0; k < N; k++)
            for (int j = 0; j < N; j++)
                hRef[i * N + j] += hA[i * N + k] * hB[k * N + j];

    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytes);
    cudaMalloc(&dB, bytes);
    cudaMalloc(&dC, bytes);
    cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice);

    const int sizes[] = {4, 8, 16, 32};
    printf("%-6s %5s %10s %9s  %s\n", "kernel", "block", "avg (ms)", "GFLOPS", "check");

    for (int tiled = 0; tiled < 2; tiled++) {
        for (int s = 0; s < 4; s++) {
            int bs = sizes[s];
            cudaMemset(dC, 0, bytes);              // a broken kernel can't pass on old data

            launch(tiled, bs, dA, dB, dC);         // warm-up, not timed
            cudaDeviceSynchronize();
            cudaError_t err = cudaGetLastError();
            if (err != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(err));
                return 1;
            }

            double total = 0.0;
            for (int r = 0; r < RUNS; r++) {
                double t0 = now();
                launch(tiled, bs, dA, dB, dC);
                cudaDeviceSynchronize();           // wait for the kernel before stopping the clock
                total += now() - t0;
            }
            double avg_ms = total / RUNS * 1e3;
            double gflops = 2.0 * N * N * N / (avg_ms / 1e3) / 1e9;

            cudaMemcpy(hC, dC, bytes, cudaMemcpyDeviceToHost);
            int ok = 1;
            for (int i = 0; i < N * N; i++)
                if (!(fabsf(hC[i] - hRef[i]) <= 1e-3f * fmaxf(1.0f, fabsf(hRef[i]))))
                    ok = 0;

            printf("%-6s %5d %10.3f %9.1f  %s\n", tiled ? "tiled" : "naive", bs,
                   avg_ms, gflops, ok ? "PASS" : "FAIL");
        }
    }
    return 0;
}
