#ifndef EXAMPLE_H
#define EXAMPLE_H

#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// 定义结构体，方便理解，事实上还是所有的值构成一个一维数组
// Maxtrix(x, y) = elements[x * w + y]
struct Matrix {
    int w;
    int h;
    float* elements;
};

void addFunc(float* a, float* b, float* c, int n);

void mulFunc(Matrix* a, Matrix* b, Matrix* c, const dim3 blocks, const dim3 grids);

#endif  // EXAMPLE_H