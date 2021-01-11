#include <iostream>

#include "example.h"

using namespace std;

int main(int args, char** argv) {
    // cuda编程测试 mul函数
    // 1. 申请host内存
    int w = 1 << 10;
    int h = 1 << 10;
    size_t size = w * h * sizeof(float);
    Matrix* A = new Matrix(), *B = new Matrix(), *C = new Matrix();
    float* ae = (float*)malloc(size);
    float* be = (float*)malloc(size);
    float* ce = (float*)malloc(size);
    for (int i = 0; i  < h * w; ++i) {
        ae[i] = 2;
        be[i] = 2;
    }
    A->w = w;
    A->h = h;
    B->w = h;
    B->h = w;
    C->w = A->w;
    C->h = B->h;
    // 2. 申请device内存
    Matrix* dA, *dB, *dC;
    float* d_ae, *d_be, *d_ce;
    cudaMalloc((void**)&dA, sizeof(Matrix));
    cudaMalloc((void**)&dB, sizeof(Matrix));
    cudaMalloc((void**)&dC, sizeof(Matrix));
    cudaMalloc((void**)&d_ae, size);
    cudaMalloc((void**)&d_be, size);
    cudaMalloc((void**)&d_ce, size);
    // 3. 将host的数据拷贝到device上
    cudaMemcpy((void*)d_ae, (void*)ae, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_be, (void*)be, size, cudaMemcpyHostToDevice);
    A->elements = d_ae;
    B->elements = d_be;
    C->elements = d_ce;
    cudaMemcpy((void*)dA, (void*)A, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dB, (void*)B, sizeof(Matrix), cudaMemcpyHostToDevice);
    cudaMemcpy((void*)dC, (void*)C, sizeof(Matrix), cudaMemcpyHostToDevice);
    // 4. 执行封装的host函数
    dim3 blocks(32, 32);
    dim3 grids((A->w + blocks.x - 1) / blocks.x, (B->h + blocks.y - 1) / blocks.y);
    mulFunc(dA, dB, dC, blocks, grids);
    // cudaDeviceSynchronize();
    // 5. 将device上的结果拷贝到host上，打印查看结果
    cudaMemcpy((void*)C, (void*)dC, sizeof(Matrix), cudaMemcpyDeviceToHost);
    cudaMemcpy((void*)ce, (void*)C->elements, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i ) {
        cout << "res: " << ce[i] << " " << C->w << " " << C->h << endl;
    }
    // 6. 释放host和device上申请的内存
    cudaFree(d_ae);
    cudaFree(d_be);
    cudaFree(d_ce);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(ae);
    free(be);
    free(ce);
    return 0;
}
