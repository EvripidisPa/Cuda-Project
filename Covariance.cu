#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

/* Problem size */
#define M 1024
#define N 1024

#define FLOAT_N 3214212.01

void init_arrays(double* data)
{
	int i, j;

	for (i = 0; i < (M); i++) {
		for (j = 0; j < (N); j++) {
			data[i*(N)+j] = ((double)(i + 1)*(j + 1)) / M;
		}
	}
}

void covariance(double* data, double* symmat, double* mean)
{
	int	i, j, j1, j2;

	/* Determine mean of column vectors of input data matrix */
	for (j = 1; j < (M + 1); j++) {
		mean[j] = 0.0;
		for (i = 1; i < (N + 1); i++) {
			mean[j] += data[i*(M + 1) + j];
		}
		mean[j] /= FLOAT_N;
	}

	/* Center the column vectors. */
	for (i = 1; i < (N + 1); i++) {
		for (j = 1; j < (M + 1); j++) {
			data[i*(M + 1) + j] -= mean[j];
		}
	}

	/* Calculate the m * m covariance matrix. */
	for (j1 = 1; j1 < (M + 1); j1++) {
		for (j2 = j1; j2 < (M + 1); j2++) {
			symmat[j1*(M + 1) + j2] = 0.0;
			for (i = 1; i < N + 1; i++) {
				symmat[j1*(M + 1) + j2] += data[i*(M + 1) + j1] * data[i*(M + 1) + j2];
			}
			symmat[j2*(M + 1) + j1] = symmat[j1*(M + 1) + j2];
		}
	}
}

int main(int argc, char *argv[])
{
	double		*data;
	double		*symmat;
	double		*mean;

	FILE * fp;
	FILE * mine;

	fp = fopen("basic.txt", "w+");
	mine = fopen("mine.txt", "w+");

	data = (double*)malloc( M * N  * sizeof(double));

	symmat = (double*)malloc((M + 1)*(M + 1) * sizeof(double));
	
	mean = (double*)malloc((M + 1) * sizeof(double));

	init_arrays(data);

	//covariance(data, symmat, mean);

	free(data);
	free(symmat);
	free(mean);

	return 0;
}