#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int **alloc_matrix(int n) {
    int **M = malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++)
        M[i] = malloc(n * sizeof(int));
    return M;
}

void fill_matrix(int **M, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            M[i][j] = rand();
}

/* Reference kernel: copied from the handout */
void matmul(int **A, int **B, int **C, int rowsA, int colsA, int colsB) {
    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsB; j++) {
            C[i][j] = 0;
            for (int k = 0; k < colsA; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

/* Tiled kernel: assumes bs divides n evenly */
void matmul_tiled(int **A, int **B, int **C, int n, int bs) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            C[i][j] = 0;

    for (int ii = 0; ii < n; ii += bs)
        for (int jj = 0; jj < n; jj += bs)
            for (int kk = 0; kk < n; kk += bs)
                for (int i = ii; i < ii + bs; i++)
                    for (int j = jj; j < jj + bs; j++)
                        for (int k = kk; k < kk + bs; k++)
                            C[i][j] += A[i][k] * B[k][j];
}

int matrices_equal(int **X, int **Y, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            if (X[i][j] != Y[i][j])
                return 0;
    return 1;
}

void free_matrix(int **M, int n) {
    for (int i = 0; i < n; i++)
        free(M[i]);
    free(M);
}


int main(int argc, char *argv[]) {

    int bs = atoi(argv[1]);
    int n = 1024;

    int **A = alloc_matrix(n);
    int **B = alloc_matrix(n);
    int **C_ref = alloc_matrix(n);
    int **C_tiled = alloc_matrix(n);

    srand(10);
    fill_matrix(A, n);
    fill_matrix(B, n);

    struct timespec start, end;

    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul(A, B, C_ref, n, n, n);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double t_baseline = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    clock_gettime(CLOCK_MONOTONIC, &start);
    matmul_tiled(A, B, C_tiled, n, bs);
    clock_gettime(CLOCK_MONOTONIC, &end);

    double t_tiled = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("tile size: %d\n", bs);
    printf("baseline:  %.4f s\n", t_baseline);
    printf("tiled:     %.4f s\n", t_tiled);
    if (matrices_equal(C_ref, C_tiled, n)) {
        printf("check: PASS\n");
    } else {
        printf("check: FAIL\n");
    }
    

    free_matrix(A, n);
    free_matrix(B, n);
    free_matrix(C_ref, n);
    free_matrix(C_tiled, n);
    return 0;
}
