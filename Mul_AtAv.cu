/*TO DO 
	No major issues , the code could be optimised a little bit more. Specifically in the init of the array section.
	1** Remove the debugging printf commands.
	
	*/

#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))		//CuBlas column-major storage , the array index of a matrix element in row “i” and column “j” can be computed via this macro 

/*Problem size*/

#define HEIGHT 4                                 //  number  of rows of a
#define WIDTH  3								 //  number  of  columns  of a

#ifndef M_PI
#define M_PI 3.14159
#endif

void init_array(double *x, double *y, double *A)
{
	int i, j;

	for (i = 0; i < WIDTH; i++) x[i] = i * M_PI;						// x={i * M_PI;}^T
	for (i = 0; i < HEIGHT; i++) y[i] = 0.0f;							// y={0...0}^T


	for (i = 0; i < HEIGHT; i++) {										
		for (j = 0; j < WIDTH; j++) {
			A[IDX2C(i, j, HEIGHT)] = ((double)i*(j)) / HEIGHT;
		}
	}
}

int main()
{
	cudaError_t  cuda_status;                              //  cudaMalloc  status
	cublasStatus_t  cuBl_stat;                               //  CUBLAS  functions  status
	cublasHandle_t  handle;                             //  CUBLAS  context

	size_t sizeA = WIDTH * HEIGHT * sizeof(double);
	size_t size_x = WIDTH * sizeof(double);
	size_t size_y = HEIGHT * sizeof(double);

	int i, j;

	double		*A;										//	A - HEIGHT x WIDTH matrix on the host
	double		*x;										//	x -  WIDTH vector the host
	double		*y; 									//	y -  HEIGHT vector the host

	A = (double*)malloc(sizeA);
	if (A == NULL) {
		fprintf(stderr, "malloc() Failed");
		return -1;
	}

	x = (double*)malloc(size_x);
	if (x == NULL) {
		fprintf(stderr, "malloc() Failed");
		return -1;
	}

	y = (double*)malloc(size_y);
	if (y == NULL) {
		fprintf(stderr, "malloc() Failed");
		return -1;
	}

	init_array(x, y , A);

	printf("A:\n");

	for (i = 0; i < HEIGHT; i++) {										
		for (j = 0; j < WIDTH; j++) {
			printf("%f ", A[IDX2C(i, j, HEIGHT)]);
		}
		printf("\n");
	}

	printf(" Now normally \n");

	for (i = 0; i < HEIGHT; i++) {
		for (j = 0; j < WIDTH; j++) {
			printf("%f ", A[i*WIDTH + j] );
		}
		printf("\n");
	}

	printf("And vectors x , y\n");

	for (i = 0; i < WIDTH; i++) {
		printf("%f ", x[i]);
	}

	printf("\n");

	for (i = 0; i < HEIGHT; i++) {
		printf("%f ", y[i]);
	}

	printf("\n");

	/*-------------"Kernel" Preparation ------------*/

	double* d_A;
	double* d_x;
	double* d_y;

	cuda_status = cudaMalloc((void **)&d_A, sizeA);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}

	cuda_status = cudaMalloc((void **)&d_x, size_x);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}

	cuda_status = cudaMalloc((void **)&d_y, size_y);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "cudaMalloc Failed!");
		goto Error;
	}

	/*----------- cuBLAS parameter setup ----------*/

	cuBl_stat = cublasCreate(&handle);
	cuBl_stat = cublasSetMatrix(HEIGHT, WIDTH, sizeof(*A), A, HEIGHT, d_A, HEIGHT);          //cp a->d_a
	cuBl_stat = cublasSetVector(WIDTH, sizeof(*x), x, 1, d_x, 1);							 //cp x->d_x
	cuBl_stat = cublasSetVector(HEIGHT, sizeof(*y), y, 1, d_y, 1);							 //cp y->d_y

	double alpha = 1.0;
	double beta = 0.0;

	// matrix -vector  multiplication:    d_y = alpha * d_a * d_x + beta * d_y
	// d_A - HEIGHT x WIDTH   matrix; d_x - WIDTH-vector , d_y - HEIGHT-vector;
	// alpha ,beta - scalars

	cuBl_stat = cublasDgemv(handle, CUBLAS_OP_N, HEIGHT, WIDTH, &alpha, d_A, HEIGHT, d_x, 1, &beta, d_y, 1);

	cuBl_stat = cublasGetVector(HEIGHT, sizeof(*y), d_y, 1, y, 1);             //copy d_y -> y

	printf("y after  Dgemv ::\n");

	for (j = 0; j < HEIGHT; j++)
	{
		printf("%f ", y[j]);                                     //  print y after  dgemv
	}
	printf("\n");

	/*------ Now that we have calculated y = A * x we shall input that y in another Dgemv this time with A^T ---------*/

	cuBl_stat = cublasDgemv(handle, CUBLAS_OP_T, HEIGHT, WIDTH, &alpha, d_A, HEIGHT, d_y, 1, &beta, d_y, 1);

	cuBl_stat = cublasGetVector(WIDTH, sizeof(*y), d_y, 1, y, 1);

	printf("y after second Dgemv ::\n");

	for (j = 0; j < WIDTH; j++)
	{
		printf("%f ", y[j]);                                     //  print y after  dgemv
	}
	printf("\n");

Error:
	cudaFree(d_A);
	cudaFree(d_x);
	cudaFree(d_y);
	cublasDestroy(handle);                                     //  destroy  CUBLAS  context
	free(A);                                                    // free  host  memory
	free(x);                                                    // free  host  memory
	free(y);                                                    // free  host  memory

	return 0;
}
	