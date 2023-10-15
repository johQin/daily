#include<cuda_runtime.h>
#include"common/common.h"
#include<stdio.h>
#include <time.h>
#include<unistd.h>

// 初始化需要计算的数据
int initialData(float *head, int nElem){
    time_t t;
    srand((unsigned) time(&t));
    for(int i=0; i<nElem; i++){
        head[i] = (float) (rand() & 0xFF) / 10.0f;
    }
    printf("\n");
    return 1;

}

// 矩阵加法计算
__global__ void sumArrayOnGpu(float *a,float *b, float *c, const int N){
    int i = threadIdx.x + blockIdx.x * blockDim.x;      // 线程在线程块的索引threadIdx.x，线程所在的线程块索引blockIdx.x，每个线程块的线程数blockDim.x（在这里是32）
    if(i<N){
        c[i] = a[i] + b[i];
    }

}
int main(int argc, char **argv){
    int nDeviceNumber = 0;
    // 检查设备
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if(error != cudaSuccess || nDeviceNumber == 0){
        printf("No CUDA compatable GPU found\n");
        return -1;
    }

    // 设置某个设备为工作设备
    int dev = 0;
    error = ErrorCheck(cudaSetDevice(dev), __FILE__, __LINE__);
    if(error != cudaSuccess ){
        printf("fail to set GPU 0 for computing\n");
        return -1;
    }

    //分配主机内存
    int nElem = 1<<14;
    size_t  nBytes = nElem * sizeof(float);
    float *h_a, *h_b,*gpuRef;
    h_a = (float *) malloc(nBytes);
    h_b = (float *) malloc(nBytes);
    gpuRef = (float *) malloc(nBytes);
    if(h_a==NULL || h_b == NULL || gpuRef == NULL){
        printf("memory allocate fail\n");
        if(h_a != NULL) free(h_a);
        if(h_b != NULL) free(h_b);
        if(gpuRef != NULL) free(gpuRef);
        return -1;
    }else{
        printf("memory allocate successfully\n");
    }

    //初始化主机内存
    initialData(h_a,nElem);
    sleep(1);
    initialData(h_b,nElem);
    memset(gpuRef, 0, nBytes);      //存放结果

    //分配GPU内存
    float *d_a, *d_b,*d_c;
    cudaMalloc((float **)&d_a, nBytes);
    cudaMalloc((float **)&d_b, nBytes);
    cudaMalloc((float **)&d_c, nBytes);
    if(d_a==NULL || d_b == NULL || d_c == NULL){
        printf("fail to allocate GPU memory");
        free(h_a);
        free(h_b);
        free(gpuRef);
        return -1;
    }else{
        printf("GPU memory allocate successfully\n");
    }

    // 将data 从host内存复制到gpu内存中去
    if(
            cudaSuccess == cudaMemcpy(d_a,h_a,nBytes,cudaMemcpyHostToDevice) &&
            cudaSuccess == cudaMemcpy(d_b,h_b,nBytes,cudaMemcpyHostToDevice) &&
            cudaSuccess == cudaMemcpy(d_c,gpuRef,nBytes,cudaMemcpyHostToDevice)
            ){
        printf("Successfully copy data from CPU to GPU\n");
    }else{
        printf("fail to copy data from CPU to GPU\n");
    }

    //设置线程布局
    dim3 block(32);     //每个block的线程数
    dim3 grid(nElem/32);    // block的数量
    printf("Execution configure <<<%d, %d>>>, total element: %d",grid.x,block.x,nElem);
    double dTime_Begin = GetCPUSecond();
    // 调用核函数
    sumArrayOnGpu<<<grid, block>>>(d_a, d_b, d_c, nElem);
    cudaDeviceSynchronize();
    double dTime_End = GetCPUSecond();

    // 将计算结果从gpu中拷贝到cpu中来，内部有同步机制，会等到核函数计算完毕之后，再拷贝
    cudaMemcpy(gpuRef,d_c,nBytes,cudaMemcpyDeviceToHost);
    // 打印前20个数据
    for(int i=0; i<20; i++){
        printf("idx=%d, matrix_A:%.2f, matrix_B:%.2f, result=%.2f\n", i+1, h_a[i], h_b[i], gpuRef[i]);
    }

    printf("Element size: %d, Matrix add time Elapse is: %.5f\n", nElem, dTime_End - dTime_Begin);

    // 释放资源
    free(h_a);
    free(h_b);
    free(gpuRef);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaDeviceReset();
    return 0;
}