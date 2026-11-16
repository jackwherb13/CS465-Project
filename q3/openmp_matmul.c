
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

#define N 1024

int A[N][N], B[N][N];
unsigned int C[N][N];

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Enter the number of threads\n");
        return 1;
    }

    int num_threads = atoi(argv[1]);

    if (num_threads < 1 || num_threads > 64) {
        printf("Choose between 1 and 64 threads\n");
        return 1;
    }

    srand(10);

    // Fill both matrices with random numbers
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i][j] = rand();
            B[i][j] = rand();
        }
    }

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Divide the rows between threads
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            unsigned int sum = 0;

            for (int k = 0; k < N; k++) {
                sum += (unsigned int)A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("OpenMP with %d threads: %.6f seconds\n",
           num_threads, time_taken);

    // Print a result to check the calculation
    printf("C[0][0] = %u\n", C[0][0]);

    return 0;
}
