#include<cuda_runtime.h>
#include"common/common.h"
#include <stdio.h>
#include<unistd.h>
#include <time.h>




__global__ void sumArrayOnGpu1D(int *A,int *B,int *C,const int nx, const int ny){

    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if(ix < nx){
        for(int iy =0;iy<ny;iy++){
            int idx = iy*nx + ix;
            C[idx] = A[idx]+ B[idx];
        }
    }
}

int main(int argc, char **argv) {
    //从命令行获取block的布局
//    if(argc != 2) return -1;
//    int block_x = atoi(argv[1]);
    // 手动设置block的布局
    int block_x = 32;

    int nDeviceNumber = 0;
    // 检查设备
    cudaError_t error = ErrorCheck(cudaGetDeviceCount(&nDeviceNumber), __FILE__, __LINE__);
    if (error != cudaSuccess || nDeviceNumber == 0) {
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
    int nx = 1<<14;     //16384
    int ny = 1<<14;
    int nxy = nx*ny;
    int  nBytes = nxy * sizeof(int);
    int *h_a, *h_b,*gpuRef;
    h_a = (int *) malloc(nBytes);
    h_b = (int *) malloc(nBytes);
    gpuRef = (int *) malloc(nBytes);

    //初始化主机内存
    for(int i =0;i<nxy;i++){
        h_a[i] = i;
        h_b[i] = i+1;
    }
    memset(gpuRef, 0, nBytes);      //存放结果

    //分配GPU内存
    int *d_a, *d_b,*d_c;
    cudaMalloc((void **)&d_a, nBytes);
    cudaMalloc((void **)&d_b, nBytes);
    cudaMalloc((void **)&d_c, nBytes);


    // 将data 从host内存复制到gpu内存中去
    cudaMemcpy(d_a,h_a,nBytes,cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,h_b,nBytes,cudaMemcpyHostToDevice);

    //设置线程布局
    dim3 block(block_x,1);
    // 这样的布局可以以使得每行数据可以在一个block里计算；
    dim3 grid((nx + block.x -1)/block.x, 1);
    printf("thread config:grid<%d, %d>, block<%d, %d>\n", grid.x, grid.y, block.x, block.y);

    // 调用内核
    double dTime_Begin = GetCPUSecond();
    // 调用核函数
    sumArrayOnGpu1D<<<grid, block>>>(d_a, d_b, d_c,nx,ny);
    // 同步
    cudaDeviceSynchronize();
    double dTime_End = GetCPUSecond();
    printf("Element size: %d, Matrix add time Elapse is: %.5f\n", nxy, dTime_End - dTime_Begin);
    cudaMemcpy(gpuRef,d_c,nBytes,cudaMemcpyDeviceToHost);
    for(int i=0; i<10; i++){
        printf("idx=%d, matrix_A:%d, matrix_B:%d, result=%d\n", i+1, h_a[i], h_b[i], gpuRef[i]);
    }

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