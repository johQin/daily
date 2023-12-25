# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目信息
# 指定语言信息
project(trt_demo LANGUAGES CXX CUDA)

include(CMakePrintHelpers)		# 这是一个打印帮助工具

# https://blog.csdn.net/qq_33642342/article/details/116459742
# 报这个错：CMP0104: CMAKE_CUDA_ARCHITECTURES now detected for NVCC, empty CUDA_ARCHITECTURES not allowed的时候，加下面这一段
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 70 75 80)
endif(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)


# 添加CMAKE_MODULE_PATH，否则找不到FindTensorRT.cmake
list (APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# 寻找TensorRT库，如果使用deb包安装tensortRT的话，可以使用这种方式
#find_package(TensorRT REQUIRED HINTS /usr/local/TensorRT-8.6.1.6)
#if (TensorRT_FOUND)
#    message(STATUS "Found TensorRT ${TensorRT_VERSION} in ${TensorRT_ROOT_DIR}")
#    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
#    message(STATUS "TensorRT include files: ${TensorRT_INCLUDE_DIRS}")
#else()
#    message(FATAL_ERROR "Cannot find TensorRT")
#endif()

set(CUDA_INCLUDE_DIRS "/usr/local/cuda/include" CACHE PATH "TensorRT include directory")
# 如果使用tar包的方式解压的TensorRT，需要指定对应的变量
set(TensorRT_ROOT "/usr/local/TensorRT-8.6.1.6" CACHE PATH "TensorRT root directory")
set(TensorRT_INCLUDE_DIRS "${TensorRT_ROOT}/include" CACHE PATH "TensorRT include directory")

set(TensorRT_LIB_NAME nvinfer nvinfer_plugin nvonnxparser nvinfer_builder_resource)
set(TensorRT_LIBRARIES)
FOREACH (tlib IN LISTS TensorRT_LIB_NAME)
    unset(TRT_LIB)
    find_library(TRT_LIB NAMES ${tlib} HINTS "${TensorRT_ROOT}" PATH_SUFFIXES lib lib64 lib/x64 NO_CACHE)
    if(TRT_LIB)
        list(APPEND TensorRT_LIBRARIES ${TRT_LIB})
    ENDIF ()
ENDFOREACH ()
CMAKE_PRINT_VARIABLES(TensorRT_LIBRARIES)


file(GLOB CPPS
        ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp
        )
set(INCLUDE_DIRS)
list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/utils)


# 添加可执行文件
add_executable(build src/build.cpp)

# 头文件
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${INCLUDE_DIRS})
# 链接库
target_link_libraries(build PRIVATE ${TensorRT_LIBRARIES} ${CPPS})

# 添加可执行文件
add_executable(runtime src/runtime.cu)
# 头文件
target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIRS} ${CUDA_INCLUDE_DIRS} ${INCLUDE_DIRS})

# 链接库
target_link_libraries(runtime PRIVATE ${TensorRT_LIBRARIES} ${CPPS})

