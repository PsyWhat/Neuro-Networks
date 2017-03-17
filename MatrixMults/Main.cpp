#include "Matrix.h"
#include "conio.h"


void main()
{
	Matrix *a = MatrixGetNew( 65 , 75 );
	Matrix *b = MatrixGetNew( 75 , 10 );
	Matrix *cpures = NULL;


	for ( int i = 0; i < a->rows; ++i )
	{
		for ( int j = 0; j < a->colums; ++j )
		{
			MatrixSetElem( a , i , j , i+j );
		}
	}
	for ( int i = 0; i < b->rows; ++i )
	{
		for ( int j = 0; j < b->colums; ++j )
		{
			MatrixSetElem( b , i , j , i+j );
		}
	}

	printf( "Starting GPU mult.\n" );
	Matrix *res = MatrixMultiply( a , b );
	printf( "Finished GPU mult.\n" );


	printf( "Starting CPU mult.\n" );
	cpures = MatrixCPUMultithreadBlockMultiply( a , b );
	printf( "Finished CPU mult.\n" );

	double error = 0.0;
	for ( int i = 0; i < res->rows; ++i )
	{
		for ( int j = 0; j < res->colums; ++j )
		{
			error += abs( res->values[i + j*res->rows] - cpures->values[i + j * cpures->rows] );
		}
	}
	printf( "Error rate: %f.\n",error);

	/*printf( "Matrix A:\n" );
	MatrixPrint( a );
	printf( "Matrix B:\n" );
	MatrixPrint( b );
	printf( "Matrix Res:\n" );
	MatrixPrint( res );*/

	FreeMatrix( res );
	FreeMatrix( cpures );
	FreeMatrix( a );
	FreeMatrix( b );

	_getch();
}