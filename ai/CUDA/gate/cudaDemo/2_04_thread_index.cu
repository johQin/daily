#include <stdio.h>

__global__ void helloFromGPU(){
    printf("gridDim:x= %d, y=%d, z=%d, lockDim: x= %d, y=%d, z=%d, Current threadIdx: x=%d, y=%d, z=%d\n",
           gridDim.x, gridDim.y, gridDim.z,
           blockDim.x, blockDim.y, blockDim.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}
int main(int argc, char ** argv){
    printf("hello world from cpu\n");
    // grid 里面包含2 x 2的4个block
    dim3 grid;
    grid.x = 2;
    grid.y = 2;
    // 每个block里面又包含了2 x 2的4个thread
    dim3 block;
    block.x = 2;
    block.y = 2;
    // <<<1, 10>>>这里的1指定了网格grid里block的个数为1，10指定了每个block包含的线程数为10
    helloFromGPU<<<grid, block>>>();
    // 释放设备资源
    cudaDeviceReset();
    return 0;
}