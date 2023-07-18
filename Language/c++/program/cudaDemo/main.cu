#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

int main() {

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cout << "使用GPU device " << dev << ": " << deviceProp.name << std::endl;
    std::cout << "SM的数量：" << deviceProp.multiProcessorCount << std::endl;
    std::cout << "每个线程块的共享内存大小：" << deviceProp.sharedMemPerBlock / 1024.0 << " KB" << std::endl;
    std::cout << "每个线程块的最大线程数：" << deviceProp.maxThreadsPerBlock << std::endl;
    std::cout << "每个EM的最大线程数：" << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "每个SM的最大线程束数：" << deviceProp.maxThreadsPerMultiProcessor / 32 << std::endl;
    std::cout << "Hello, World!" << std::endl;
    return 0;
}
