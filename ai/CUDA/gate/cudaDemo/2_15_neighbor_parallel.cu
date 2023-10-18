#include<stdio.h>
#include<unistd.h>
#include<cuda_runtime.h>
#include<device_functions.h>
#include"common/common.h"
__global__ void reduceNeighbored(int *g_idata, int *g_odata,unsigned int n){
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    //边界检查
    if(idx > n) return;

    for(int stride = 1; stride < blockDim.x;stride *=2){
        if((tid % (2 *stride)) == 0){
            idata[tid] += idata[tid + stride];
        }
        // 线程块同步函数
        __syncthreads();
    }

    if(tid == 0) g_odata[blockIdx.x] = idata[0];

}
int main(){
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

    int size = 1 << 24;
    int blockSize = 512;

    dim3 block(blockSize,1);
    dim3 grid((size + block.x - 1) / block.x, 1);

    size_t bytes = size * sizeof(int);
    int* h_idata = (int *) malloc(bytes);
    int* h_odata = (int *) malloc(grid.x*sizeof(int));
    int* tmp = (int *) malloc(bytes);

    for(int i =0; i<size;i++){
        // 使数字最大是255
        h_idata[i]= (int) (rand() & 0xFF);
    }
    memcpy(tmp, h_idata, bytes);
    double iStart, iElaps;
    int gpu_sum = 0;

    int *d_idata = NULL;
    int *d_odata = NULL;
    cudaMalloc((void **) &d_idata,bytes);
    cudaMalloc((void **) &d_odata,grid.x * sizeof(int));

    cudaMemcpy(d_idata,h_idata,bytes,cudaMemcpyHostToDevice);
    iStart = GetCPUSecond();
    reduceNeighbored<<<grid, block>>>(d_idata,d_odata,size);
    cudaDeviceSynchronize();
    iElaps = GetCPUSecond() - iStart;

    cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
    gpu_sum = 0;

    for(int i =0;i<grid.x;i++) gpu_sum += h_odata[i];
    printf("gpu Neighbored add Elapse is: %.5f sec, gpu_sum: %d <<<grid %d, block %d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);

    cudaDeviceReset();
    return 0;




}