#include<cuda_runtime.h>
#include"common/common.h"
#include <stdio.h>
#include<unistd.h>

void printMatrix(int *m, const int nx, const int ny){
    for(int iy = 0; iy<ny; iy++){
        for(int ix =0; ix<nx; ix++){
            printf("%3d\t", m[iy*nx + ix]);
        }
        printf("\n");
    }
}

__global__ void printThreadIndex(int *A,const int nx, const int ny){

   int ix = threadIdx.x + blockIdx.x * blockDim.x;
   int iy = threadIdx.y + blockIdx.y * blockDim.y;
   unsigned int idx = iy * nx + ix;
   printf("thread_id (%d, %d) block_id (%d, %d) coordinate (%d, %d) global index"
          "%2d ival %2d\n", threadIdx.x, threadIdx.y,blockIdx.x, blockIdx.y,ix,iy,idx, A[idx]);
}

int main(int argc, char **argv) {
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

    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(int);

    //分配主机内存
    int *h_a;
    h_a = (int *) malloc(nBytes);
    for(int i =0; i<nxy;i++){   //初始化主机内存
        h_a[i]= i;
    }
    printMatrix(h_a,nx,ny);

    //分配GPU内存
    int *d_a;
    error = ErrorCheck(cudaMalloc((void **) &d_a, nBytes), __FILE__, __LINE__);
    if(error != cudaSuccess){
        printf("fail to allocate memory for GPU\n");
        free(h_a);
        return -1;
    }

    // 将data 从host内存复制到gpu内存中去
    error = ErrorCheck(cudaMemcpy(d_a,h_a, nBytes, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    if(error != cudaSuccess){
        printf("fail to copy data from host to GPU\n");
        free(h_a);
        cudaFree(d_a);
        return -1;
    }

    //设置线程布局
    dim3 block(4,2);
    // 这样的布局可以以使得每行数据可以在一个block里计算；
    dim3 grid((nx + block.x -1)/block.x, (ny + block.y -1)/block.y);

    // 调用内核
    printThreadIndex<<<grid, block>>>(d_a,nx,ny);

    //释放资源
    cudaFree(d_a);
    free(h_a);
    cudaDeviceReset();
    return 0;
}