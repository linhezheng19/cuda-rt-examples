#include "example.h"

__global__ void add(float* a, float* b, float* c, int n) {
    int thread_idx = threadIdx.x + blockIdx.x * blockDim.x;  // 该线程所在位置
    int stride = blockDim.x * gridDim.x;  // 该线程所在的线程块一共有多少个线程，
                                          // gridDim.x个block，每个block有blockDim.x个thread
                                          // 如果一个线程块不够，就使用下一个线程块
    for (int i = thread_idx; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}

__device__ float getVal(Matrix* a, int row, int col) {
    // 相当于cpp中的inline函数，只不过是在device中调用
    return a->elements[row * a->w + col];
}

__device__ void setVal(Matrix* c, int row, int col, float val) {
    c->elements[row * c->w + col] = val;
}

__global__ void mul(Matrix* a, Matrix* b, Matrix* c) {
    float c_val = 0.f;
    // 这里计算row和col指的是c中的元素的位置，而不是a或b中元素的位置
    // 要根据row和col去找ab中的元素进行计算，也就是说这个线程并行是
    // 并行计算c的结果，该线程负责c中该位置的结果，而c的结果通过row
    // 和col去a、b中寻找
    int row = blockDim.y * blockIdx.y + threadIdx.y;  // 一个block有多少行，乘上第几个block，加上在该block的第几个thread
    int col = blockDim.x * blockIdx.x + threadIdx.x;  // 同上，计算列号
    for (int i = 0; i < a->w; ++i) {
        // a, b 一个按列取，一个按行取
        c_val += getVal(a, row, i) * getVal(b, i, col);
    }
    setVal(c, row, col, c_val);
}

void addFunc(float* a, float* b, float* c, int n) {
    // 初始化kernel
    dim3 blocks(512);  // 一维
    dim3 grids((n + blocks.x - 1) / blocks.x);  // 计算n个数需要多少个block
    add<<< grids, blocks >>>(a, b, c, n);
}

void mulFunc(Matrix* a, Matrix* b, Matrix* c, const dim3 blocks, const dim3 grids) {
    // 调用kernel
    mul<<< grids, blocks >>>(a, b, c);
}
