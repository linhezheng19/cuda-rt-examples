#include <iostream>

#include "example.h"

using namespace std;

int main(int args, char** argv) {
    // cuda编程测试 add函数
    // 1. 申请host内存
    int n = 1 << 20;  // 假设有这么多个数据要计算
    size_t size = sizeof(float) * n;
    float *a, *b, *c;
    a = static_cast<float*>(malloc(size));
    b = static_cast<float*>(malloc(size));
    c = static_cast<float*>(malloc(size));
    for (int i = 0; i < n / 2; ++i) {  // 初始化一半的数据
        a[i] = i;
        b[i] = i;
    }
    // cout << "aa: " << a[0] << a[1] << endl;
    // 2. 申请device内存
    float *da, *db, *dc;
    cudaMalloc((void**)&da, size);  // cuda分配内存用的指针的指针，是一个指针参数
    cudaMalloc((void**)&db, size);
    cudaMalloc((void**)&dc, size);
    // 3. 将host的数据拷贝到device上
    cudaMemcpy((void*)da, (void*)a, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void*)db, (void*)b, size, cudaMemcpyHostToDevice);
    // 4. 执行封装的host函数
    addFunc(da, db, dc, n);
    // 5. 将device上的结果拷贝到host上，打印查看结果
    cudaMemcpy((void*)c, (void*)dc, size, cudaMemcpyDeviceToHost);
    for (int i = 0; i < 10; ++i ) {
        cout << "a + b = " << c[i] << endl;
    }
    // 6. 释放host和device上申请的内存
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    free(a);
    free(b);
    free(c);
// ------------------------------------------------------===================>
    // cuda multiple, 这里涉及到带指针的结构体，所以采用cudaMallocManaged进行统一内存管理
    // 不需要人为进行拷贝与复制，人为复制在test2.cpp中
    int w = 1 << 10;
    int h = 1 << 10;
    Matrix *A, *B, *C;
    cudaMallocManaged((void**)&A, sizeof(Matrix));
    cudaMallocManaged((void**)&B, sizeof(Matrix));
    cudaMallocManaged((void**)&C, sizeof(Matrix));
    size_t s = w * h * sizeof(float);
    cudaMallocManaged((void**)&A->elements, s);
    cudaMallocManaged((void**)&B->elements, s);
    cudaMallocManaged((void**)&C->elements, s);

    A->w = w;
    A->h = h;
    B->w = w;
    B->h = h;
    C->w = w;
    C->h = h;
    for (int i = 0; i < h * w; ++i){
        A->elements[i] = 2;
        B->elements[i] = 2;
    }

    dim3 blocks(32, 32);
    dim3 grids((w + blocks.x - 1) / blocks.x, (h + blocks.y - 1) / blocks.y);  // -1也可以不用
    mulFunc(A, B, C, blocks, grids);
    cudaDeviceSynchronize();  // 统一内存需要手动同步，手动复制不需要，因为在cudaMemcpy中会进行一次同步
    for (int i = 0; i < 10; ++i ) {
        cout << "res: " << C->elements[i] << " " << C->w << " " << C->h << endl;
    }

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
