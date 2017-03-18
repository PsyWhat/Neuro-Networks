#include "Matrix.h"



Matrix* MatrixGetNew( int rows , int columns )
{
	Matrix *res = (Matrix *)malloc( sizeof( Matrix ) );
	res->colums = columns;
	res->rows = rows;
	res->values = (double*)malloc( sizeof( double ) * rows * columns );

	return res;
}

void MatrixNull( Matrix *a )
{
	for ( int i = 0; i < (a->colums * a->rows); ++i )
	{
		a->values[i] = 0.0;
	}
}

Matrix* MatrixCopy( Matrix *a )
{
	Matrix *res = MatrixGetNew( a->rows , a->colums );
	for ( int i = 0; i < res->rows; ++i )
	{
		for ( int j = 0; j < res->colums; ++j )
		{
			res->values[i + j*res->rows] = a->values[i + j * res->rows];
		}
	}
	return res;
}

Matrix* MatrixTransposed( Matrix *a, int numThreads )
{
	Matrix *res = MatrixCPUMultithreadTranspose( a, numThreads );
	return res;
}

void FreeMatrix( Matrix *m )
{
	if ( m != NULL )
	{
		if ( m->values != NULL )
		{
			free( m->values );
		}
		
		free( m );
	}
}

double MatrixGetElem( Matrix *a , int row , int column )
{
	return a->values[row + column * a->rows];
}

void MatrixSetElem( Matrix *a , int row , int column , double value )
{
	a->values[row + column * a->rows] = value;
}

Matrix* MatrixColConcat( Matrix *a , Matrix *b )
{
	Matrix *res = MatrixGetNew(  a->rows + b->rows, 1 );
	for ( int i = 0; i < res->rows; ++i )
	{
		if ( i < a->rows )
		{
			res->values[i] = a->values[i];
		} else
		{
			res->values[i] = b->values[i - a->rows];
		}
	}
	return res;
}

Matrix* MatrixRowConcat( Matrix *a , Matrix *b )
{
	Matrix *res = MatrixGetNew( 1, a->colums + b->colums );
	for ( int i = 0; i < res->colums; ++i )
	{
		if ( i < a->colums )
		{
			res->values[i] = a->values[i];
		} else
		{
			res->values[i] = b->values[i - a->colums];
		}
	}
	return res;
}

Matrix* MatrixMultiply( Matrix *a , Matrix *b , int numThreads )
{
	Matrix *res = CudaMatMult( a , b );
	if ( res == NULL )
	{
		res = MatrixCPUMultithreadBlockMultiply( a , b , numThreads );
	}
	return res;
}

Matrix* MatrixMultiplyDouble( Matrix *a , double d , int numThreads /*= 16 */ )
{
	Matrix *res;
	res = CudaMultDouble( a , d );
	if ( res == NULL )
	{
		res = MatrixCPUMultithreadMultDouble( a , d , numThreads );
	}
	return res;
}

Matrix* MatrixAddition( Matrix *a , Matrix *b , int numThreads )
{

	Matrix *res;
	res = CudaAddMats( a , b );
	if ( res == NULL )
	{
		res = MatrixCPUMultithreadAdd( a , b , numThreads );
	}
	return res;
}

Matrix* MatrixHadamardProduct( Matrix *a , Matrix *b, int numThreads )
{
	Matrix *res;
	res = CudaHadamardProduct( a , b );
	if ( res == NULL )
	{
		res = MatrixCPUMultithreadHadamardProduct( a , b , numThreads );
	}
	return res;
}

void MatrixPrint( Matrix *a )
{
	for ( int i = 0; i < a->rows; ++i )
	{
		for ( int j = 0; j < a->colums; ++j )
		{
			printf( "%+5.3f " , a->values[i + a->rows * j] );
		}
		printf( "\n" );
	}
}


Matrix* MatrixCPUMultithreadBlockMultiply( Matrix *a , Matrix *b, int maxThreads)
{
	using std::thread;
	using std::queue;
	queue<thread*> threads;
	Matrix *res = NULL;
	res = MatrixGetNew( a->rows , b->colums );

	int rows = a->rows;
	int cols = b->colums;
	if ( rows > cols )
	{
		for ( int i = 0; i < rows; ++i )
		{
			thread *t = new thread( [i,&cols,&res,&a, &b]() {
				for ( int j = 0; j < cols; ++j )
				{
					res->values[i + j * res->rows] = 0.0;
					for ( int m = 0; m < a->colums; ++m )
					{
						res->values[i + j * res->rows] += a->values[i + m * a->rows] * b->values[m + j * b->rows];
					}
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}
	} else
	{

		for ( int j = 0; j < cols; ++j )
		{
			thread *t = new thread( [j , &rows , &res , &a , &b](){
				for ( int i = 0; i < rows; ++i )
				{
					res->values[i + j * res->rows] = 0.0;
					for ( int m = 0; m < a->colums; ++m )
					{
						res->values[i + j * res->rows] += a->values[i + m * a->rows] * b->values[m + j * b->rows];
					}
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}

	}

	return res;
}

Matrix* MatrixCPUMultithreadHadamardProduct( Matrix *a , Matrix *b , int maxThreads /*= 16 */ )
{
	using std::thread;
	using std::queue;
	queue<thread*> threads;
	Matrix *res = NULL;
	res = MatrixGetNew( a->rows , b->colums );

	int rows = a->rows;
	int cols = b->colums;
	if ( rows > cols )
	{
		for ( int i = 0; i < rows; ++i )
		{
			thread *t = new thread( [i , &cols , &res , &a , &b](){
				for ( int j = 0; j < cols; ++j )
				{
					res->values[i + j * res->rows] = a->values[i + j * a->rows] * b->values[i + j * b->rows];
					
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}
	} else
	{

		for ( int j = 0; j < cols; ++j )
		{
			thread *t = new thread( [j , &rows , &res , &a , &b](){
				for ( int i = 0; i < rows; ++i )
				{
					res->values[i + j * res->rows] = a->values[i + j * a->rows] * b->values[i + j * b->rows];
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}

	}

	return res;
}

Matrix* MatrixCPUMultithreadAdd( Matrix *a , Matrix *b , int maxThreads /*= 16 */ )
{
	using std::thread;
	using std::queue;
	queue<thread*> threads;
	Matrix *res = NULL;
	res = MatrixGetNew( a->rows , b->colums );

	int rows = a->rows;
	int cols = b->colums;
	if ( rows > cols )
	{
		for ( int i = 0; i < rows; ++i )
		{
			thread *t = new thread( [i , &cols , &res , &a , &b](){
				for ( int j = 0; j < cols; ++j )
				{
					res->values[i + j * res->rows] = a->values[i + j * a->rows] + b->values[i + j * b->rows];

				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}
	} else
	{

		for ( int j = 0; j < cols; ++j )
		{
			thread *t = new thread( [j , &rows , &res , &a , &b](){
				for ( int i = 0; i < rows; ++i )
				{
					res->values[i + j * res->rows] = a->values[i + j * a->rows] + b->values[i + j * b->rows];
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}

	}

	return res;
}

Matrix* MatrixCPUMultithreadSub( Matrix *a , Matrix *b , int maxThreads /*= 16 */ )
{

	using std::thread;
	using std::queue;
	queue<thread*> threads;
	Matrix *res = NULL;
	res = MatrixGetNew( a->rows , b->colums );

	int rows = a->rows;
	int cols = b->colums;
	if ( rows > cols )
	{
		for ( int i = 0; i < rows; ++i )
		{
			thread *t = new thread( [i , &cols , &res , &a , &b](){
				for ( int j = 0; j < cols; ++j )
				{
					res->values[i + j * res->rows] = a->values[i + j * a->rows] - b->values[i + j * b->rows];

				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}
	} else
	{

		for ( int j = 0; j < cols; ++j )
		{
			thread *t = new thread( [j , &rows , &res , &a , &b](){
				for ( int i = 0; i < rows; ++i )
				{
					res->values[i + j * res->rows] = a->values[i + j * a->rows] - b->values[i + j * b->rows];
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}

	}

	return res;
}

Matrix* MatrixCPUMultithreadMultDouble( Matrix *a , double d , int maxThreads /*= 16 */ )
{
	using std::thread;
	using std::queue;
	queue<thread*> threads;
	Matrix *res = NULL;
	res = MatrixGetNew( a->rows , a->colums );

	int rows = a->rows;
	int cols = a->colums;
	if ( rows > cols )
	{
		for ( int i = 0; i < rows; ++i )
		{
			thread *t = new thread( [i ,&d, &cols , &res , &a ](){
				for ( int j = 0; j < cols; ++j )
				{
					res->values[i + j * res->rows] = a->values[i + j * a->rows]  * d;

				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}
	} else
	{

		for ( int j = 0; j < cols; ++j )
		{
			thread *t = new thread( [j ,&d, &rows , &res , &a](){
				for ( int i = 0; i < rows; ++i )
				{
					res->values[i + j * res->rows] = a->values[i + j * a->rows] * d;
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}

	}

	return res;
}

Matrix* MatrixCPUMultithreadTranspose( Matrix *a , int maxThreads /*= 16 */ )
{

	using std::thread;
	using std::queue;
	queue<thread*> threads;
	Matrix *res = NULL;
	res = MatrixGetNew( a->colums, a->rows );

	int rows = a->rows;
	int cols = a->colums;
	if ( rows > cols )
	{
		for ( int i = 0; i < rows; ++i )
		{
			thread *t = new thread( [i , &cols , &res , &a](){
				for ( int j = 0; j < cols; ++j )
				{
					res->values[j + i * res->rows] = a->values[i + j * a->rows];

				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}
	} else
	{

		for ( int j = 0; j < cols; ++j )
		{
			thread *t = new thread( [j , &rows , &res , &a](){
				for ( int i = 0; i < rows; ++i )
				{
					res->values[j + i * res->rows] = a->values[i + j * a->rows];
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}

	}

	return res;
}

Matrix* MatrixGetRow( Matrix *a , int row )
{
	Matrix *res = NULL;

	res = MatrixGetNew( 1 , a->colums );

	for(int i = 0; i < a->colums;++i )
	{
		res->values[i] = a->values[row + a->rows * i];
	}

	return res;
}

Matrix* MatrixGetColumn( Matrix *a , int column )
{

	Matrix *res = NULL;

	res = MatrixGetNew( a->rows , 1 );

	for ( int i = 0; i < a->rows; ++i )
	{
		res->values[i] = a->values[i + a->rows * column];
	}

	return res;
}

EXPORT Matrix* MatrixSubstraction( Matrix *a , Matrix *b , int maxThreads /*= 16 */ )
{

	Matrix *res;
	res = CudaSubMats( a , b );
	if ( res == NULL )
	{
		res = MatrixCPUMultithreadSub( a , b , maxThreads );
	}
	return res;
}

Matrix* MatrixNegateMultithread( Matrix *a , int maxThreads /*= 16 */ )
{

	using std::thread;
	using std::queue;
	queue<thread*> threads;
	Matrix *res = NULL;
	res = MatrixGetNew(  a->rows, a->colums );

	int rows = a->rows;
	int cols = a->colums;
	if ( rows > cols )
	{
		for ( int i = 0; i < rows; ++i )
		{
			thread *t = new thread( [i , &cols , &res , &a](){
				for ( int j = 0; j < cols; ++j )
				{
					res->values[i + j * res->rows] = - a->values[i + j * a->rows];

				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}
	} else
	{

		for ( int j = 0; j < cols; ++j )
		{
			thread *t = new thread( [j , &rows , &res , &a](){
				for ( int i = 0; i < rows; ++i )
				{
					res->values[i + j * res->rows] = - a->values[i + j * a->rows];
				}
			} );
			threads.push( t );
			while ( threads.size() > maxThreads )
			{
				thread *w = threads.front();
				if ( w->joinable() )
				{
					w->join();
				} else
				{
					threads.pop();
					delete w;
				}
			}
		}
		while ( threads.size() > 0 )
		{
			thread *w = threads.front();
			if ( w->joinable() )
			{
				w->join();
			} else
			{
				threads.pop();
				delete w;
			}
		}

	}

	return res;
}

Matrix* MatrixNegate( Matrix *a , int maxThreads /*= 16 */ )
{
	Matrix *res = MatrixNegateMultithread( a , maxThreads );
	return res;
}

