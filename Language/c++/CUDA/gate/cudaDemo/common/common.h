//
// Created by buntu on 2023/10/8.
//

#ifndef CUDADEMO_COMMON_H
#define CUDADEMO_COMMON_H

#include<sys/time.h>
#include<cuda_runtime.h>
#include<stdio.h>

cudaError_t ErrorCheck(cudaError_t status, const char * filename, int lineNumber){
    if(status != cudaSuccess){
        printf("CUDA RUNTIME API ERROR: \r\ncode=%d, name=%s, decription=%s\r\nfile=%s, line=%d\r\n", status,
               cudaGetErrorName(status), cudaGetErrorString(status), filename, lineNumber);
    }
    return status;
}

#endif //CUDADEMO_COMMON_H
