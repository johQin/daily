#include <stdio.h>

__global__ void helloFromGPU(){
    printf("hello world from GPU\n");
}
int main(int argc, char ** argv){
    printf("hello world from cpu\n");

    // <<<1, 10>>>这里的1指定了网格grid里block的个数为1，10指定了每个block包含的线程数为10
    helloFromGPU<<<1, 10>>>();
    // 释放设备资源
    cudaDeviceReset();
    return 0;
}