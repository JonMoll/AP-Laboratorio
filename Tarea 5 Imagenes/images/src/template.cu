#include <stdio.h>
#include <stdlib.h>

#include "lodepng.h"

void encodeOneStep(const char* filename, const unsigned char* image, unsigned width, unsigned height)
{
	unsigned error = lodepng_encode32_file(filename, image, width, height);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));
}

__global__ void imageBNKernel(unsigned char* d_image, int h, int w)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;
	int r, g, b;

	if((Row < h) && (Col < w)){
		r = d_image[4 * w * Row + 4 * Col + 0];
		g = d_image[4 * w * Row + 4 * Col + 1];
		b = d_image[4 * w * Row + 4 * Col + 2];

		d_image[4 * w * Row + 4 * Col + 0] = 0;
		d_image[4 * w * Row + 4 * Col + 1] = 0;
		d_image[4 * w * Row + 4 * Col + 2] = 0;
		d_image[4 * w * Row + 4 * Col + 3] = (int)(r*0.21 + g*0.71 + b*0.07);
	}
}

void imageBN(unsigned char* image, int h, int w)
{
	unsigned char *d_image;
	int size = (h * w) * 4;

	cudaMalloc((void **) &d_image, size);
	cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);

	int blockSize = 16;

	dim3 DimGrid(ceil((h-1)/blockSize+1), ceil((w-1)/blockSize+1), 1);
	dim3 DimBlock(blockSize, blockSize, 1);
	imageBNKernel<<< DimGrid, DimBlock >>>(d_image, h, w);

	cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);

	cudaFree(d_image);
}

__global__ void imageBlurKernel(unsigned char* d_image, int h, int w)
{
	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;

	int blurSize = 8;

	Row = Row * blurSize;
	Col = Col * blurSize;

	int r, g, b;
	int p_r = 0;
	int p_g = 0;
	int p_b = 0;
	int i, j;

	if((Row+blurSize < h) && (Col+blurSize < w)){
		for(i = 0; i < blurSize; i++)
			for(j = 0; j < blurSize; j++){
				r = d_image[4 * w * (Row+j) + 4 * (Col+i) + 0];
				g = d_image[4 * w * (Row+j) + 4 * (Col+i) + 1];
				b = d_image[4 * w * (Row+j) + 4 * (Col+i) + 2];

				p_r += r;
				p_g += g;
				p_b += b;
			}

		p_r = p_r / (blurSize * blurSize);
		p_g = p_g / (blurSize * blurSize);
		p_b = p_b / (blurSize * blurSize);

		for(i = 0; i < blurSize; i++)
			for(j = 0; j < blurSize; j++){
				d_image[4 * w * (Row+j) + 4 * (Col+i) + 0] = p_r;
				d_image[4 * w * (Row+j) + 4 * (Col+i) + 1] = p_g;
				d_image[4 * w * (Row+j) + 4 * (Col+i) + 2] = p_b;
			}
	}
}

void imageBlur(unsigned char* image, int h, int w)
{
	unsigned char *d_image;
	int size = (h * w) * 4;

	cudaMalloc((void **) &d_image, size);
	cudaMemcpy(d_image, image, size, cudaMemcpyHostToDevice);

	int blockSize = 16;

	dim3 DimGrid(ceil((h-1)/blockSize+1), ceil((w-1)/blockSize+1), 1);
	dim3 DimBlock(blockSize, blockSize, 1);
	imageBlurKernel<<< DimGrid, DimBlock >>>(d_image, h, w);

	cudaMemcpy(image, d_image, size, cudaMemcpyDeviceToHost);

	cudaFree(d_image);
}

int main()
{
	printf("Begin\n");

	unsigned error;
	unsigned char* image;
	unsigned width, height;

	const char* filename = "/home/jonathan/Cuda/parrot.png";
	const char* filenameOut = "out.png";

	error = lodepng_decode32_file(&image, &width, &height, filename);
	if(error) printf("error %u: %s\n", error, lodepng_error_text(error));

	printf("Width %d\nHeight %d\n", width, height);

	unsigned x, y;
	int r, g, b, a;

	for(y = 0; y < height; y++)
		for(x = 0; x < width; x++){
			r = image[4 * width * y + 4 * x + 0];
			g = image[4 * width * y + 4 * x + 1];
			b = image[4 * width * y + 4 * x + 2];
			a = image[4 * width * y + 4 * x + 3];

			printf("OLD PIXEL[%d, %d]:(%d, %d, %d, %d)\n", x, y, r, g, b, a);
		}

	imageBN(image, height, width);
	//imageBlur(image, height, width);

	for(y = 0; y < height; y++)
		for(x = 0; x < width; x++){
			r = image[4 * width * y + 4 * x + 0];
			g = image[4 * width * y + 4 * x + 1];
			b = image[4 * width * y + 4 * x + 2];
			a = image[4 * width * y + 4 * x + 3];

			printf("NEW PIXEL[%d, %d]:(%d, %d, %d, %d)\n", x, y, r, g, b, a);
		}

	encodeOneStep(filenameOut, image, width, height);

	free(image);

	printf("End\n");

	return 0;
}
