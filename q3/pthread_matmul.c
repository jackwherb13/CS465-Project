
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>

#define N 1024

int A[N][N], B[N][N];
unsigned int C[N][N];

int num_threads;

// Each thread calculates part of the matrix
void *multiply(void *arg) {
    int thread_id = *(int *)arg;

    int start = thread_id * N / num_threads;
    int end = (thread_id + 1) * N / num_threads;

    for (int i = start; i < end; i++) {
        for (int j = 0; j < N; j++) {
            unsigned int sum = 0;

            for (int k = 0; k < N; k++) {
                sum += (unsigned int)A[i][k] * B[k][j];
            }

            C[i][j] = sum;
        }
    }

    return NULL;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Enter the number of threads\n");
        return 1;
    }

    num_threads = atoi(argv[1]);

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

    pthread_t threads[64];
    int ids[64];

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);

    // Create threads
    for (int i = 0; i < num_threads; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, multiply, &ids[i]);
    }

    // Wait for threads to finish
    for (int i = 0; i < num_threads; i++) {
        pthread_join(threads[i], NULL);
    }

    clock_gettime(CLOCK_MONOTONIC, &end);

    double time_taken = (end.tv_sec - start.tv_sec) +
                        (end.tv_nsec - start.tv_nsec) / 1000000000.0;

    printf("Pthreads with %d threads: %.6f seconds\n",
           num_threads, time_taken);

    // Print a result to check the calculation
    printf("C[0][0] = %u\n", C[0][0]);

    return 0;
}
