#pragma once

#include <cstdlib>
#include <stdio.h>
#include "CUDAFunctions.h"
#include "MatrixStruct.h"
#include <queue>
#include <thread>

#define EXPORT __declspec(dllexport)

extern "C"
{
	EXPORT Matrix* MatrixGetNew( int rows , int columns );

	EXPORT void MatrixNull( Matrix *a );

	EXPORT Matrix* MatrixCopy( Matrix *a );

	EXPORT Matrix* MatrixTransposed( Matrix *a , int numThreads );

	EXPORT void FreeMatrix( Matrix *m );

	EXPORT double MatrixGetElem( Matrix *a , int row , int column );

	EXPORT void MatrixSetElem( Matrix *a , int row , int column , double value );

	EXPORT Matrix* MatrixColConcat( Matrix *a , Matrix *b );

	EXPORT Matrix* MatrixRowConcat( Matrix *a , Matrix *b );

	EXPORT Matrix* MatrixMultiply( Matrix *a , Matrix *b , int maxThreads = 16 );

	EXPORT Matrix* MatrixMultiplyDouble( Matrix *a , double d , int maxThreads = 16 );

	EXPORT Matrix* MatrixAddition( Matrix *a , Matrix *b , int maxThreads = 16 );

	EXPORT Matrix* MatrixHadamardProduct( Matrix *a , Matrix *b , int maxThreads = 16 );

	EXPORT Matrix* MatrixGetRow( Matrix *a , int row );

	EXPORT Matrix* MatrixGetColumn( Matrix *a , int column );

}

//void MatrixPrint( Matrix *a );



Matrix* MatrixCPUMultithreadBlockMultiply( Matrix *a , Matrix *b, int maxThreads = 16 );


Matrix* MatrixCPUMultithreadHadamardProduct( Matrix *a , Matrix *b , int maxThreads = 16 );

Matrix* MatrixCPUMultithreadAdd( Matrix *a , Matrix *b , int maxThreads = 16 );

Matrix* MatrixCPUMultithreadMultDouble( Matrix *a , double d , int maxThreads = 16 );


Matrix* MatrixCPUMultithreadTranspose( Matrix *a , int maxThreads = 16 );