

#pragma once


#include "CUDAFunctions.h"








#define BLOCK_SIZE 32







__global__ void matBlockMultKernel( Matrix a , Matrix b , Matrix res )
{
	int blockRow = blockIdx.y;
	int blockColumn = blockIdx.x;


	double Cvalue = 0.0;

	int row = threadIdx.y;
	int col = threadIdx.x;

	int rowm = row + blockRow * blockDim.y;
	int colm = col + blockColumn * blockDim.x;

	int addition = (a.colums % BLOCK_SIZE == 0) ? 0 : 1;

	for ( int m = 0; m < (a.colums / BLOCK_SIZE) + addition; ++m )
	{

		__shared__ double As[BLOCK_SIZE][BLOCK_SIZE];
		__shared__ double Bs[BLOCK_SIZE][BLOCK_SIZE];


		int rown = (m * blockDim.y + row);
		int coln = (m * blockDim.x + col);

		if ( rowm < a.rows &&  coln < a.colums )
		{
			As[row][col] = a.values[rowm + a.rows * coln];
		} else
		{
			As[row][col] = 0.0;
		}

		if ( rown < b.rows && colm < b.colums )
		{
			Bs[row][col] = b.values[rown + b.rows * colm];
		} else
		{
			Bs[row][col] = 0.0;
		}

		__syncthreads();

		for ( int e = 0; e < BLOCK_SIZE; ++e )
		{
			Cvalue += As[row][e] * Bs[e][col];
		}

		__syncthreads();

	}
	if ( rowm < res.rows && colm < res.colums )
	{
		res.values[rowm + res.rows * colm] = Cvalue;
	}
}


__host__
Matrix* CudaMatMult( Matrix *A , Matrix *B , int device )
{
	Matrix *res = NULL;

	Matrix a , b , r;
	cudaError_t err;
	int deviceCount;

	a.rows = A->rows;
	a.colums = A->colums;
	a.values = NULL;

	b.rows = B->rows;
	b.colums = B->colums;
	b.values = NULL;

	r.rows = A->rows;
	r.colums = B->colums;
	r.values = NULL;


	if ( cudaGetDeviceCount( &deviceCount ) < device )
	{
		err = cudaError::cudaErrorInvalidDevice;
		goto FREE;
	}

	err = cudaSetDevice( device );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&(a.values)) , sizeof( double ) * a.colums * a.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&b.values) , sizeof( double ) * b.colums * b.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&r.values) , sizeof( double ) * r.colums * r.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

	err = cudaMemcpy( (void*)a.values , A->values , sizeof( double ) * a.colums * a.rows , cudaMemcpyHostToDevice );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMemcpy( b.values , B->values , sizeof( double ) * b.colums * b.rows , cudaMemcpyHostToDevice );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

	dim3 blockSize( BLOCK_SIZE , BLOCK_SIZE );
	dim3 numBlocks( (r.colums / BLOCK_SIZE) + ((r.colums % BLOCK_SIZE == 0) ? 0 : 1) , (r.rows / BLOCK_SIZE) + ((r.rows % BLOCK_SIZE == 0) ? 0 : 1) );

	matBlockMultKernel << <numBlocks , blockSize >> > (a , b , r);

	err = cudaPeekAtLastError();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaGetLastError();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaDeviceSynchronize();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	//checkCudaErrors( cudaPeekAtLastError() );
	//checkCudaErrors( cudaDeviceSynchronize() );


	res = (Matrix*)malloc( sizeof( Matrix ) );
	res->colums = r.colums;
	res->rows = r.rows;
	res->values = NULL;
	res->values = (double*)malloc( sizeof( double ) * res->colums * res->rows );

	err = cudaMemcpy( res->values , r.values , sizeof( double ) * (res->colums) * (res->rows) , cudaMemcpyDeviceToHost );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

FREE:

	cudaFree( a.values );
	cudaFree( b.values );
	cudaFree( r.values );

	if ( err != cudaSuccess && res != NULL )
	{
		if ( res->values != NULL )
		{
			free( res->values );
		}
		free( res );
		res = NULL;
	}


	return res;
}





__global__
void CudaAddMatsKernell( Matrix a , Matrix b , Matrix r )
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if ( row < r.rows && col < r.colums )
	{
		r.values[row + col * r.rows] = a.values[row + col * a.rows] + b.values[row + col * b.rows];
	}

}



__host__
Matrix* CudaAddMats( Matrix *A , Matrix *B , int device /*= 0 */ )
{
	Matrix *res = NULL;

	Matrix a , b , r;
	cudaError_t err;
	int deviceCount;

	a.rows = A->rows;
	a.colums = A->colums;
	a.values = NULL;

	b.rows = B->rows;
	b.colums = B->colums;
	b.values = NULL;

	r.rows = A->rows;
	r.colums = A->colums;
	r.values = NULL;


	if ( cudaGetDeviceCount( &deviceCount ) < device )
	{
		err = cudaError::cudaErrorInvalidDevice;
		goto FREE;
	}

	err = cudaSetDevice( device );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&(a.values)) , sizeof( double ) * a.colums * a.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&b.values) , sizeof( double ) * b.colums * b.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&r.values) , sizeof( double ) * r.colums * r.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

	err = cudaMemcpy( (void*)a.values , A->values , sizeof( double ) * a.colums * a.rows , cudaMemcpyHostToDevice );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMemcpy( b.values , B->values , sizeof( double ) * b.colums * b.rows , cudaMemcpyHostToDevice );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

	dim3 blockSize( BLOCK_SIZE , BLOCK_SIZE );
	dim3 numBlocks( (r.colums / BLOCK_SIZE) + ((r.colums % BLOCK_SIZE == 0) ? 0 : 1) , (r.rows / BLOCK_SIZE) + ((r.rows % BLOCK_SIZE == 0) ? 0 : 1) );

	CudaAddMatsKernell << <numBlocks , blockSize >> > (a , b , r);

	err = cudaPeekAtLastError();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaGetLastError();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaDeviceSynchronize();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	//checkCudaErrors( cudaPeekAtLastError() );
	//checkCudaErrors( cudaDeviceSynchronize() );


	res = (Matrix*)malloc( sizeof( Matrix ) );
	res->colums = r.colums;
	res->rows = r.rows;
	res->values = NULL;
	res->values = (double*)malloc( sizeof( double ) * res->colums * res->rows );

	err = cudaMemcpy( res->values , r.values , sizeof( double ) * (res->colums) * (res->rows) , cudaMemcpyDeviceToHost );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

FREE:

	cudaFree( a.values );
	cudaFree( b.values );
	cudaFree( r.values );

	if ( err != cudaSuccess && res != NULL )
	{
		if ( res->values != NULL )
		{
			free( res->values );
		}
		free( res );
		res = NULL;
	}


	return res;
}







__global__
void CudaHadamardKernell( Matrix a , Matrix b , Matrix r )
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if ( row < r.rows && col < r.colums )
	{
		r.values[row + col * r.rows] = a.values[row + col * a.rows] * b.values[row + col * b.rows];
	}

}


__host__
Matrix* CudaHadamardProduct( Matrix *A , Matrix *B , int device /*= 0 */ )
{

	Matrix *res = NULL;

	Matrix a , b , r;
	cudaError_t err;
	int deviceCount;

	a.rows = A->rows;
	a.colums = A->colums;
	a.values = NULL;

	b.rows = B->rows;
	b.colums = B->colums;
	b.values = NULL;

	r.rows = A->rows;
	r.colums = A->colums;
	r.values = NULL;


	if ( cudaGetDeviceCount( &deviceCount ) < device )
	{
		err = cudaError::cudaErrorInvalidDevice;
		goto FREE;
	}

	err = cudaSetDevice( device );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&(a.values)) , sizeof( double ) * a.colums * a.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&b.values) , sizeof( double ) * b.colums * b.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&r.values) , sizeof( double ) * r.colums * r.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

	err = cudaMemcpy( (void*)a.values , A->values , sizeof( double ) * a.colums * a.rows , cudaMemcpyHostToDevice );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMemcpy( b.values , B->values , sizeof( double ) * b.colums * b.rows , cudaMemcpyHostToDevice );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

	dim3 blockSize( BLOCK_SIZE , BLOCK_SIZE );
	dim3 numBlocks( (r.colums / BLOCK_SIZE) + ((r.colums % BLOCK_SIZE == 0) ? 0 : 1) , (r.rows / BLOCK_SIZE) + ((r.rows % BLOCK_SIZE == 0) ? 0 : 1) );

	CudaHadamardKernell << <numBlocks , blockSize >> > (a , b , r);

	err = cudaPeekAtLastError();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaGetLastError();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaDeviceSynchronize();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	//checkCudaErrors( cudaPeekAtLastError() );
	//checkCudaErrors( cudaDeviceSynchronize() );


	res = (Matrix*)malloc( sizeof( Matrix ) );
	res->colums = r.colums;
	res->rows = r.rows;
	res->values = NULL;
	res->values = (double*)malloc( sizeof( double ) * res->colums * res->rows );

	err = cudaMemcpy( res->values , r.values , sizeof( double ) * (res->colums) * (res->rows) , cudaMemcpyDeviceToHost );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

FREE:

	cudaFree( a.values );
	cudaFree( b.values );
	cudaFree( r.values );

	if ( err != cudaSuccess && res != NULL )
	{
		if ( res->values != NULL )
		{
			free( res->values );
		}
		free( res );
		res = NULL;
	}


	return res;
}


__global__
void CudaMultDoubleKernell( Matrix a , double d , Matrix r )
{
	int row = threadIdx.y + blockDim.y * blockIdx.y;
	int col = threadIdx.x + blockDim.x * blockIdx.x;

	if ( row < r.rows && col < r.colums )
	{
		r.values[row + col * r.rows] = a.values[row + col * a.rows] * d;
	}
}

__host__
Matrix* CudaMultDouble( Matrix *A , double d, int device )
{
	Matrix *res = NULL;

	Matrix a  , r;
	cudaError_t err;
	int deviceCount;

	a.rows = A->rows;
	a.colums = A->colums;
	a.values = NULL;


	r.rows = A->rows;
	r.colums = A->colums;
	r.values = NULL;


	if ( cudaGetDeviceCount( &deviceCount ) < device )
	{
		err = cudaError::cudaErrorInvalidDevice;
		goto FREE;
	}

	err = cudaSetDevice( device );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaMalloc( (void**)(&(a.values)) , sizeof( double ) * a.colums * a.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	
	err = cudaMalloc( (void**)(&r.values) , sizeof( double ) * r.colums * r.rows );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

	err = cudaMemcpy( (void*)a.values , A->values , sizeof( double ) * a.colums * a.rows , cudaMemcpyHostToDevice );
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	

	dim3 blockSize( BLOCK_SIZE , BLOCK_SIZE );
	dim3 numBlocks( (r.colums / BLOCK_SIZE) + ((r.colums % BLOCK_SIZE == 0) ? 0 : 1) , (r.rows / BLOCK_SIZE) + ((r.rows % BLOCK_SIZE == 0) ? 0 : 1) );

	CudaMultDoubleKernell <<<numBlocks , blockSize >> > (a , d , r);

	err = cudaPeekAtLastError();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaGetLastError();
	if ( err != cudaError::cudaSuccess )
		goto FREE;
	err = cudaDeviceSynchronize();
	if ( err != cudaError::cudaSuccess )
		goto FREE;

	res = (Matrix*)malloc( sizeof( Matrix ) );
	res->colums = r.colums;
	res->rows = r.rows;
	res->values = NULL;
	res->values = (double*)malloc( sizeof( double ) * res->colums * res->rows );

	err = cudaMemcpy( res->values , r.values , sizeof( double ) * (res->colums) * (res->rows) , cudaMemcpyDeviceToHost );
	if ( err != cudaError::cudaSuccess )
		goto FREE;

FREE:

	cudaFree( a.values );
	cudaFree( r.values );

	if ( err != cudaSuccess && res != NULL )
	{
		if ( res->values != NULL )
		{
			free( res->values );
		}
		free( res );
		res = NULL;
	}


	return res;
}




