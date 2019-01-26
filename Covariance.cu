#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size */
#define M 1024
#define N 1024

#define BLOCK_SIZE_PER_DIM 16

#define FLOAT_N 3214212.01

__global__ void mean_cu(double* A, double* mean, int thread_num ) {
  extern __shared__ double A_per_blk[];
	unsigned int bx = blockIdx.x;
	unsigned int by = blockIdx.y;					
	unsigned int tx = threadIdx.x;
	unsigned int ty = threadIdx.y;					// should be 0 for this kernel

	unsigned int Row = by * thread_num + ty;	
	unsigned int i = Row * M + tx;
	unsigned int stride;

	A_per_blk[tx] = A[i];
	__syncthreads();

  // do reduction in shared mem

  for(stride = thread_num/2 ; stride > 0; stride >>= 1) {

    if (tx < stride) {
      A_per_blk[tx] += A_per_blk[tx + stride];
    }
    __syncthreads();
  }

  // write result for this block to global mem
  if(tx == 0) mean[bx] = atomicAdd(mean[bx],A_per_blk[0]);		
}


void init_arrays(double* A)
{
	int i, j;

	for (i = 0; i < (M); i++) {
		for (j = 0; j < (N); j++) {
			A[i*(N)+j] = ((double)(i + 1)*(j + 1)) / M;
		}
	}
}


void covariance(double* A, double* C, double* mean)
{
	int	i, j, j1, j2;

	/* Determine mean of column vectors of input data matrix */
	for (j = 0; j < M; j++) {
		mean[j] = 0.0;
		for (i = 0; i < N; i++) {
			mean[j] += A[i * M + j];
		}
		mean[j] /= FLOAT_N;
	}

	/* Center the column vectors. */
	for (i = 0; i < N; i++) {
		for (j = 0; j < M; j++) {
			A[i * M + j] -= mean[j];
		}
	}

	/* Calculate the m * m covariance matrix. */
	for (j1 = 0; j1 < M; j1++) {
		for (j2 = j1; j2 < M; j2++) {
			C[j1 * M + j2] = 0.0;
			for (i = 0; i < N; i++) {
				C[j1 * M + j2] += A[i * M + j1] * A[i * M + j2];
			}
			C[j2 * M + j1] = C[j1 * M + j2];
		}
	}
}

int main(int argc, char *argv[]){

	int i, j , thread_num;

	double		*A;
	double		*C;
	double		*mean;

	size_t sizeA = M * N * sizeof(double);
	size_t sizem = N * sizeof(double);
	size_t sizeC = M * M * sizeof(double);


	cudaError_t cuda_status;


	A = (double*)malloc(sizeA);
	if (A == NULL) {
		fprintf(stderr, "malloc() Failed");
		return -1;
	}

	C = (double*)calloc(M * M, sizeof(double));
	if (B == NULL) {
		fprintf(stderr, "calloc() Failed");
		return -1;
	}
	
	mean = (double*)calloc(N * sizeof(double));
	if (A == NULL) {
		fprintf(stderr, "malloc() Failed");
		return -1;
	}

	// Init array.

	init_arrays(A);

	//------Prepare the kernel ------//

	double *d_A, *d_C, *d_m;

	if (M < 512) {
		thread_num = 256;
	}
	else if (M < 1024) {
		thread_num = 512;
	}
	else {
		thread_num = 1024;
	}

	unsigned int n_Blocks_X = ((M -1) / thread_num) + 1;
	unsigned int n_Blocks_Y = N ;

	/* We can evaluate the case where our matrix has M = 1  where we could skip the mean calculation , if it wasnt for FLOAT_N */
	
	int nBlocks = n_Blocks_X * n_Blocks_Y;

	dim3 threads_per_block(thread_num, 1, 1);
	dim3 num_of_Blocks(n_Blocks_X, n_Blocks_Y, 1);
	
	size_t bytes_per_block = thread_num * 1 * 64 / 8;

	cuda_status = cudaMalloc((void **)&d_A, M * N * sizeof(double));
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}

	cuda_status = cudaMalloc((void **)&d_m, M * sizeof(double));
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}


	// Copying the data to the device 
	cuda_status = cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy Failed!");
		goto Error;
	}

	cuda_status = cudaMemcpy(d_m, mean, sizem, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy Failed!");
		goto Error;
	}

	/*--------- Kernel Call for mean ----------*/

	mean_cu <<< num_of_Blocks, threads_per_block >>> (d_A, d_m, thread_num);


	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "convolution_with_cuda launch failed: %s\n", cudaGetErrorString(cuda_status));
		goto Error;
	}

	/* cudaDeviceSynchronize waits for the kernel to finish, and returns any errors encountered during the launch. */

	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching convolution_with_cuda!\n", cuda_status);
		goto Error;
	}

	//Copying the data back and freeing the allocated space.

	cuda_status = cudaMemcpy(mean, d_m, sizem, cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy Failed!");
		goto Error;
	}


Error:
	cudaFree(d_A);
	cudaFree(d_C);
	cudaFree(d_m);

	/*-----------------Serial code our basis----------------------*/

	init_arrays(A);

	covariance(A, C, mean);

	/* ----------Debugging ------------*/

	FILE * init_array;
	FILE * out_mean;
	FILE * out_covariance;

	init_array = fopen("init_array.txt", "w+");
	out_mean = fopen("mean.txt", "w+");
	out_covariance = fopen("my_cova.txt", "w+");

	/* Debugging array initialization*/

	for (i = 0; i < M; i++) {
		for (j = 0; j < N; j++) {
			fprintf(init_array, "%f ", A[i*(N)+j]);
		}
		fprintf(init_array, "\n");
	}

	/* Debugging covariance outputs.*/

	for (j = 0; j < M; j++) {
		fprintf(out_mean, "%f\n", mean[j]);
	}

	for (i = 0; i < M; i++) {
		for (j = i; j < N; j++) {
			fprintf(out_covariance, "%f ", C[j * M + i]);
		}
		fprintf(out_covariance, "\n");
	}

	/*Cleaning up*/

	free(A);
	free(C);
	free(mean);

	return 0;
}