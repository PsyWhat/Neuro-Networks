#pragma once

#include <stdio.h>
#include <stdlib.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "MatrixStruct.h"


__host__
Matrix* CudaMatMult( Matrix *A , Matrix *B , int device = 0 );

__host__
Matrix* CudaAddMats( Matrix *A , Matrix *B, int device = 0 );

__host__
Matrix* CudaHadamardProduct( Matrix *A , Matrix *B , int device = 0 );

__host__
Matrix* CudaMultDouble( Matrix *A , double d, int device = 0 );

