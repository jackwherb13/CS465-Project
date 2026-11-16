#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <time.h>
#include <string.h>

#define SIZE 1024
#define MAX_THREADS 16

typedef struct {
	int **matrix_A;
	int **matrix_B;
	int **matrix_C;	//final answer
	// boundaries for each thread
	int first_row;	//includes first row
	int end_row;	//excludes last row
} ThreadData;


void *multiply(void *arg) {
	ThreadData *data = arg;
	
	int **matrix_A = data->matrix_A;
	int **matrix_B = data->matrix_B;
	int **matrix_C = data->matrix_C;

    	for (int i = data->first_row; i < data->end_row; i++) {
        	for (int j = 0; j < SIZE; j++) {
            		matrix_C[i][j] = 0;
	    		// k goes down col of B and across row of A
	    		// this will then add the results together
            		for (int k = 0; k < SIZE; k++) {
				// an entry in matrix C will be row i of matrix A multiplied by col j of matrix B
                		matrix_C[i][j] += matrix_A[i][k] * matrix_B[k][j];
            		}
        	}
    	}

    	return NULL;	// endung the curr workers execution
}


int **allocate(void){
	// allocating memory for the list of rows pointers
	int **matrix = malloc(SIZE * sizeof(*matrix));	//size of one row pointer = *matrix
	
	//if the alloc fails
	if (matrix == NULL){
		// perror prints out the msg and the descrip of the err
		perror("Matrix Allocation Failed.");
		exit(EXIT_FAILURE);	// same as exit(1)

	}
	// memory alloc fro each row of the answer matrix C
	for (int i =0; i<SIZE; i++){
		matrix[i] = malloc(SIZE * sizeof(*matrix[i]));
		
		//ensuring that the malloc worked
		if (matrix[i] == NULL){
			for(int j=0; j < i; j++){
				free(matrix[j]);
			
			}
			free(matrix);
			perror("Row Allocation Failed.");
			exit(EXIT_FAILURE);
		}	
	}
	
	return matrix;	// returning the pointer to matrix C answer matrix

}


// clean up func
void free_matrix(int **matrix){
	for(int i=0; i<SIZE; i++){
		free(matrix[i]);
	}
	// list of row pointers
	free(matrix);
}

// func for the matrix multiplication threads
void mm_pthreads(int **matrix_A, int **matrix_B, int **matrix_C, int num_threads){

	//arr for the threads, space for 16 max
	pthread_t threads[MAX_THREADS];	// pthreads uses pthread_t type tp recognoze a thread
	ThreadData data[MAX_THREADS];

	for(int t=0; t<num_threads; t++){
		data[t].matrix_A = matrix_A;
		data[t].matrix_B = matrix_B;
		data[t].matrix_C = matrix_C;
		data[t].first_row = t * SIZE/num_threads;
		data[t].end_row = (t+1) * SIZE/num_threads;
	}

	for (int t=0; t<num_threads; t++){
	
		int err = pthread_create( &threads[t], NULL, multiply, &data[t]);

		if(err != 0){
			fprintf(stderr, "pthread_create: %s\n", strerror(err));
			exit(EXIT_FAILURE);
		}
	}

	for( int t=0; t<num_threads; t++){
		int err = pthread_join(threads[t], NULL);
		
		if(err != 0){
			fprintf(stderr, "pthread_join: %s\n", strerror(err));
			exit(EXIT_FAILURE);
		}
	}
}

int main(int argc, char *argv[]) {
   	 // I expect one argument after the program name: the thread count.
    	if (argc != 2) {
        	fprintf(stderr, "Usage: %s <1|2|4|8|16>\n", argv[0]);
        	return EXIT_FAILURE;
    	}

    	// Convert the argument from text into a number.
    	// end points to anything left over after the number.
    	char *end;
    	long requested = strtol(argv[1], &end, 10);

    	// Reject extra characters or a thread count outside my planned tests.
    	if (*end != '\0' ||
        	(requested != 1 && requested != 2 &&
         	requested != 4 && requested != 8 && requested != 16)) {
        	fprintf(stderr, "Choose 1, 2, 4, 8, or 16 threads.\n");
        	return EXIT_FAILURE;
    	}

    	// The value is valid and small enough to store as an int.
    	int num_threads = (int)requested;

    	// Reserve memory for the two inputs and the output matrix.
    	int **matrix_A = allocate();
    	int **matrix_B = allocate();
    	int **matrix_C = allocate();

    	// Use seed 10 as required, so each run starts with the same inputs.
    	srand(10);

    	// Fill both input matrices with random integers.
    	// C is set to zero cell by cell inside the worker function.
    	for (int i = 0; i < SIZE; i++) {
        	for (int j = 0; j < SIZE; j++) {
            		matrix_A[i][j] = rand();
            		matrix_B[i][j] = rand();
        	}
    	}

    	// Store the timestamps from before and after multiplication.
    	struct timespec start, finish;

    	// Start timing after allocation and initialization are finished.
    	// CLOCK_MONOTONIC measures elapsed time without clock-setting changes.
    	if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) {
        	perror("clock_gettime");
        	return EXIT_FAILURE;
    	}

    	// Start the workers and wait until all assigned rows are finished.
    	mm_pthreads(matrix_A, matrix_B, matrix_C, num_threads);

    	// Stop timing immediately after the multiplication function returns.
    	if (clock_gettime(CLOCK_MONOTONIC, &finish) != 0) {
        	perror("clock_gettime");
        	return EXIT_FAILURE;
    	}

    	// Convert the timestamp difference into seconds.
    	// There are 1 billion nanoseconds in one second.
    	double elapsed = (finish.tv_sec - start.tv_sec) + (finish.tv_nsec - start.tv_nsec) / 1000000000.0;

    	// Print the thread count and the measured multiplication time.
    	printf("Threads: %d | Time: %.6f seconds\n", num_threads, elapsed);

    	// Show one output cell as a quick check.
    	// This alone does not verify the entire answer matrix.
    	printf("matrix_C[0][0]: %d\n", matrix_C[0][0]);

    	// Release the allocated memory after timing is finished.
    	free_matrix(matrix_A);
    	free_matrix(matrix_B);
    	free_matrix(matrix_C);

    	// Report that the program finished successfully.
    	return EXIT_SUCCESS;
}





