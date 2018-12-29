//TO-DO
//1*  Finish checking the boundaries of the blocks to ignore computations out of bounds. DONE(for the leftmsot and top)
//1** Missing bottom and right vectors.
//Global reverse ty && tx


#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE_PER_DIM 16		/*Maybe just TILE_WIDTH 16 and use just that? Though I do want a 3v3 at a time.*/
#define NUM_OF_BLOCKS 16
#define HEIGHT 32
#define WIDTH  32

__device__ __constant__ double c11 = +0.2, c21 = +0.5, c31 = -0.8, c12 = -0.3, c22 = +0.6, c32 = -0.9, c13 = +0.4, c23 = +0.7, c33 = +0.10;

__global__ void convolution_with_cuda(double* A, double* B , int num_of_Blocks) {
	
	int i, j;							

	__shared__ double A_per_blk[BLOCK_SIZE_PER_DIM][BLOCK_SIZE_PER_DIM];		/*BLOCK_SIZE_PER_DIM OR TILE WIDHT?	Edw evala 17x17 everything I need for a 16x16*/
	__shared__ double 
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	int Row = by * BLOCK_SIZE_PER_DIM + ty;
	int Col = bx * BLOCK_SIZE_PER_DIM + tx;

		A_per_blk[ty][tx] = A[Row * WIDTH + tx];
		__syncthreads();

		if ((Row < HEIGHT -1) && (Col < WIDTH -1) && (Row >= 1) && (Col >= 1) ){  // 1*
			if(tx - 1 == -1 && ty - 1 == -1){
				B = A[tx - 1][Row * WIDTH + tx]*c11	
			}	
			else{
				
			}
			B = A_per_blk[tx - 1][ty - 1]*c11
			+A_per_blk[tx][ty - 1]*c12
			+A_per_blk[tx + 1][ty - 1]*c13
			+A_per_blk[tx + 1][ty]*c23
			+A_per_blk[tx + 1][ty + 1]*c33
			+A_per_blk[tx][ty + 1]*c32
			+A_per_blk[tx - 1][ty + 1]*c31
			+A_per_blk[tx - 1][ty]*c21
			+A_per_blk[tx][ty]*c22;

			//Code to be executed
		}


	/*Want to iterate the blocks until I am done with the 3x3 matrices computations.*/

		/*	for (i = 1; i < tx_max - 1; ++i) {
				for (j = 1; j < NJ - 1; ++j) {
					B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)] + c12 * A[(i + 0)*NJ + (j - 1)] + c13 * A[(i + 1)*NJ + (j - 1)]
						+ c21 * A[(i - 1)*NJ + (j + 0)] + c22 * A[(i + 0)*NJ + (j + 0)] + c23 * A[(i + 1)*NJ + (j + 0)]
						+ c31 * A[(i - 1)*NJ + (j + 1)] + c32 * A[(i + 0)*NJ + (j + 1)] + c33 * A[(i + 1)*NJ + (j + 1)];
				}
			}
		*/
	//}
}

void init(double* A)
{
	int i, j;

	for (i = 0; i < HEIGHT; ++i) {
		for (j = 0; j < WIDTH; ++j) {
			A[i*WIDTH + j] = (double)rand() / RAND_MAX;
		}
	}
}

int main(int argc, char* argv[]) {

	int i, j;
	size_t size = WIDTH * HEIGHT * sizeof(double);

	double* A;
	double* B;

	A = (double*)malloc(size);
	if (A == NULL){
		fprintf(stderr,"malloc() Failed");
		return -1;
	}

	B = (double*)calloc(WIDTH * HEIGHT,sizeof(double));
	if (B == NULL) {
		fprintf(stderr,"calloc() Failed");
		return -1;
	}
	
	//initialize the arrays
	init(A);

	for (i = 0; i < HEIGHT; ++i) {
		for (j = 0; j < WIDTH; ++j) {
			fprintf(stdout,"A[%d,%d] = %f\n", i, j, A[i*WIDTH + j]);
		}
	}
	/************************************************Preparing the kernel *****************************************************/
	/* First the dimensions of the grid and blocks , we want them however many blocks are divided but each with 16x16 threads!
	Due to the inversed notation in the Cuda Standard ( Linear : grammes x sthles , Cuda : sthles x grammes )*/
	
	double* d_A, *d_B;

	unsigned int n_Blocks_X = ((WIDTH - 1) / BLOCK_SIZE_PER_DIM ) + 1;
	unsigned int n_Blocks_Y = ((HEIGHT - 1) / BLOCK_SIZE_PER_DIM) + 1;

	int nBlocks = n_Blocks_X * n_Blocks_Y;

	dim3 threads_per_block(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);

	dim3 num_of_Blocks(n_Blocks_X, n_Blocks_Y, 1);

	size_t bytes_per_block = 16 * 16 * 64 / 8;						/*Because we will need a 17x17 matrix of information * 64bit each (cause its a double) / by 8 (bytes)*/

	cudaError_t cuda_status = cudaMalloc((void **)&d_A, size);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
		}

	cuda_status = cudaMalloc((void **)&d_B, size);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}

	// Copying the data to the device 
	cuda_status = cudaMemcpy(d_A , A, size, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy Failed!");
		goto Error;
	}

	cuda_status = cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy Failed!");
		goto Error;
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cuda_status = cudaSetDevice(0);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}
	//Executing the kernel call 

	convolution_with_cuda<<< num_of_Blocks, threads_per_block>>>(d_A , d_B ,nBlocks);

	// Check for any errors launching the kernel
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "convolution_with_cuda launch failed: %s\n", cudaGetErrorString(cuda_status));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching convolution_with_cuda!\n", cuda_status);

		goto Error;
	}

	//Copying the data back and freeing the allocated space.

	cuda_status = cudaMemcpy(B, d_B, size, cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy Failed!");
		goto Error;
	}

	//Output the new array 
	for (i = 0; i < HEIGHT; ++i) {
		for (j = 0; j < WIDTH; ++j) {
			fprintf(stdout,"B[%d,%d] = %f\n", i, j, B[i*WIDTH + j]);
		}
	}

	Error :
	cudaFree(d_A);
	cudaFree(d_B);
	
	return 0;
}