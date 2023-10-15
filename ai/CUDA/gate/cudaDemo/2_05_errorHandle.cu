#include "common/common.h"
int main(){
    float * gpuMemory = NULL;
    ErrorCheck(cudaMalloc(&gpuMemory,sizeof(float)), __FILE__, __LINE__);
    ErrorCheck(cudaFree(gpuMemory), __FILE__, __LINE__);
    ErrorCheck(cudaFree(gpuMemory), __FILE__, __LINE__);
    ErrorCheck(cudaDeviceReset(), __FILE__, __LINE__);

    return 1;
}