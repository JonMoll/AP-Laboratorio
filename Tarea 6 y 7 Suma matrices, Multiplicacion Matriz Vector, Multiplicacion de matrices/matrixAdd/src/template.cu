/*#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>*/

#include <stdio.h>

__global__ void matrixVectMultKernel(float* A, float* B, float* C, int n)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	int elementPos = i;
	i = i * n;
	int limit = i + n;
	int j = 0;

	if(i < n*n){
		C[elementPos] = 1;

		while((i < limit) && (j < n)){
			C[elementPos] = C[elementPos] * (A[i]+B[j]);
			i++;
			j++;
		}
	}
}

void matrixVectMult(float* A, float* B, float* C, int n)
{
	int sizeMatrix = n * n * sizeof(float);
	int sizeVect = n * sizeof(float);
	float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, sizeMatrix);
	cudaMemcpy(d_A, A, sizeMatrix, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_B, sizeVect);
	cudaMemcpy(d_B, B, sizeVect, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, sizeVect);

	matrixVectMultKernel<<< ceil(n/256.0), 256 >>>(d_A, d_B, d_C, n);

	cudaMemcpy(C, d_C, sizeVect, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

__global__ void matrixAddKernel(float* A, float* B, float* C, int n)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if((Row < n) && (Col < n))
		C[Row * n + Col] = A[Row * n + Col] + B[Row * n + Col];
}

void matrixAdd(float* A, float* B, float* C, int n)
{
	int size = n * n * sizeof(float);
	float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, size);

	dim3 DimGrid(ceil((n-1)/16+1), ceil((n-1)/16+1), 1);
	dim3 DimBlock(16, 16, 1);
	matrixAddKernel<<< DimGrid, DimBlock >>>(d_A, d_B, d_C, n);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

#define TILE_WIDTH 64

int blockSize = 64;

__global__ void MatrixMulKernelV1(float* M, float* N, float* P, int Width)
{
	int Row = blockIdx.y*blockDim.y+threadIdx.y;
	int Col = blockIdx.x*blockDim.x+threadIdx.x;

	if ((Row < Width) && (Col < Width)) {
		float Pvalue = 0;

		for (int k = 0; k < Width; ++k)
			Pvalue += M[Row*Width+k]*N[k*Width+Col];

		P[Row*Width+Col] = Pvalue;
	}
}

__global__ void MatrixMulKernelV2(float* d_M, float* d_N, float* d_P, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // [TILE_WIDTH][TILE_WIDTH]
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; // [TILE_WIDTH][TILE_WIDTH]

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int ph = 0; ph < Width/TILE_WIDTH; ++ph) {
		Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];
		Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];
		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];

		__syncthreads();
	}

	d_P[Row*Width + Col] = Pvalue;
}

__global__ void MatrixMulKernelV3(float* d_M, float* d_N, float* d_P, int Width)
{
	__shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // [TILE_WIDTH][TILE_WIDTH]
	__shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; // [TILE_WIDTH][TILE_WIDTH]

	int bx = blockIdx.x; int by = blockIdx.y;
	int tx = threadIdx.x; int ty = threadIdx.y;

	int Row = by * TILE_WIDTH + ty;
	int Col = bx * TILE_WIDTH + tx;
	float Pvalue = 0;

	for (int ph = 0; ph < ceil(Width/(float)TILE_WIDTH); ++ph){
		if ((Row< Width) && (ph*TILE_WIDTH+tx)< Width)
			Mds[ty][tx] = d_M[Row*Width + ph*TILE_WIDTH + tx];

		if ((ph*TILE_WIDTH+ty)<Width && Col<Width)
			Nds[ty][tx] = d_N[(ph*TILE_WIDTH + ty)*Width + Col];

		__syncthreads();

		for (int k = 0; k < TILE_WIDTH; ++k)
			Pvalue += Mds[ty][k] * Nds[k][tx];

		__syncthreads();
	}

	if ((Row<Width) && (Col<Width))
		d_P[Row*Width + Col] = Pvalue;

}

void matrixMul1(float* A, float* B, float* C, int n)
{
	int size = n * n * sizeof(float);
	float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, size);

	dim3 DimGrid(ceil((n-1)/blockSize+1), ceil((n-1)/blockSize+1), 1);
	dim3 DimBlock(blockSize, blockSize, 1);


	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	MatrixMulKernelV1<<< DimGrid, DimBlock >>>(d_A, d_B, d_C, n);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float milisegundos = 0;
	cudaEventElapsedTime(&milisegundos, start, stop);

	printf("V1 Milisegundos: %f\n", milisegundos);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void matrixMul2(float* A, float* B, float* C, int n)
{
	int size = n * n * sizeof(float);
	float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, size);

	dim3 DimGrid(ceil((n-1)/blockSize+1), ceil((n-1)/blockSize+1), 1);
	dim3 DimBlock(blockSize, blockSize, 1);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	MatrixMulKernelV2<<< DimGrid, DimBlock >>>(d_A, d_B, d_C, n);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float milisegundos = 0;
	cudaEventElapsedTime(&milisegundos, start, stop);

	printf("V2 Milisegundos: %f\n", milisegundos);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

void matrixMul3(float* A, float* B, float* C, int n)
{
	int size = n * n * sizeof(float);
	float *d_A, *d_B, *d_C;

	cudaMalloc((void **) &d_A, size);
	cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_B, size);
	cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

	cudaMalloc((void **) &d_C, size);

	dim3 DimGrid(ceil((n-1)/blockSize+1), ceil((n-1)/blockSize+1), 1);
	dim3 DimBlock(blockSize, blockSize, 1);

	cudaEvent_t start, stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	MatrixMulKernelV3<<< DimGrid, DimBlock >>>(d_A, d_B, d_C, n);
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);

	float milisegundos = 0;
	cudaEventElapsedTime(&milisegundos, start, stop);

	printf("V3 Milisegundos: %f\n", milisegundos);

	cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
}

int main()
{
	printf("Begin\n");

	int i;
	int n = 512;

	float A[n*n];
	float B[n*n];
	float C[n*n];

	for(i = 0; i < n*n; i++)
	{
		A[i] = i;
		B[i] = i;
		C[i] = 0;
	}

	matrixMul1(A, B , C, n);
	matrixMul2(A, B , C, n);
	matrixMul3(A, B , C, n);

	/*for(i = 0; i < n*n; i++)
		printf("C[%d]: %f\n", i, C[i]);*/

	// ==================================================
/*
	float D[16];
	float E[4];
	float F[4];

	for(i = 0; i < 16; i++)
		D[i] = 2;

	for(i = 0; i < 4; i++){
		E[i] = 2;
		F[i] = 0;
	}

	matrixVectMult(D, E, F, 4);

	for(i = 0; i < 4; i++)
		printf("F[%d]: %f\n", i, F[i]);
*/
	printf("End\n");

	return 0;
}
