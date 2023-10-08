#include<cuda_runtime.h>
#include<stdio.h>

int main(int argc, char **argv){
    int deviceCount = 0;
    // 查询设备个数
    cudaGetDeviceCount(&deviceCount);
    if(deviceCount == 0){
        printf("there ard no available device(s) that support CUDA \n");
        return 1;
    }else{
        printf("Detect %d CUDA Capable device(s)\n", deviceCount);
    }
    int dev = 0,driverVersion = 0, runtimeVersion = 0;
    // 默认查询第0个设备信息
    cudaSetDevice(dev);
    // 获取设备的属性
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp,dev);
    printf("Using Device %d: \"%s\"\n", dev, deviceProp.name);

    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("CUDA Driver Version / Runtime Version %d.%d / %d.%d \n",
           driverVersion/1000, (driverVersion % 100)/10,
           runtimeVersion/1000, (runtimeVersion % 100)/10
           );

    //avail可使用的GPU显存大小，total显存总的大小
    size_t avail;
    size_t total;
    cudaMemGetInfo( &avail, &total );
    //全部显存大小
    printf("Amount of global memory: %g GB\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    //全部显存及剩余可用显存
    printf("Amount of total memory: %g GB avail memory: %g \n", total / (1024.0 * 1024.0 * 1024.0), avail / (1024.0 * 1024.0 * 1024.0));
    //计算能力：标识设备的核心架构、gpu硬件支持的功能和指令，有时也被称为“SM version”
    printf("Compute capability:     %d.%d\n", deviceProp.major, deviceProp.minor);
    //常量大小
    printf("Amount of constant memory:      %g KB\n", deviceProp.totalConstMem / 1024.0);
    //网格最大大小
    printf("Maximum grid size:  %d %d %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    //block最大
    printf("maximum block size:     %d %d %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    //SM个数
    printf("Number of SMs:      %d\n", deviceProp.multiProcessorCount);
    //每个block的共享内存大小
    printf("Maximum amount of shared memory per block: %g KB\n", deviceProp.sharedMemPerBlock / 1024.0);
    //每个SM 共享内存大小
    printf("Maximum amount of shared memory per SM:    %g KB\n", deviceProp.sharedMemPerMultiprocessor / 1024.0);
    //每个block中寄存器个数
    printf("Maximum number of registers per block:     %d K\n", deviceProp.regsPerBlock / 1024);
    //每个SM中寄存器个数
    printf("Maximum number of registers per SM:        %d K\n", deviceProp.regsPerMultiprocessor / 1024);
    //每个block最大的线程数
    printf("Maximum number of threads per block:       %d\n", deviceProp.maxThreadsPerBlock);
    //每个SM最大的线程数
    printf("Maximum number of threads per SM:          %d\n", deviceProp.maxThreadsPerMultiProcessor);

    return 1;
}