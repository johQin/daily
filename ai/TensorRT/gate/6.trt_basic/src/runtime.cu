/*
使用.cu是希望使用CUDA的编译器NVCC，会自动连接cuda库

TensorRT runtime 推理过程

1. 创建一个runtime对象
2. 反序列化生成engine：runtime ---> engine
3. 创建一个执行上下文ExecutionContext：engine ---> context

    4. 填充数据
    5. 执行推理：context ---> enqueueV2

6. 释放资源：delete

*/
#include <iostream>
#include <vector>
#include <fstream>
#include <cassert>

#include "cuda_runtime.h"
#include "NvInfer.h"

// logger用来管控打印日志级别
// TRTLogger继承自nvinfer1::ILogger
class TRTLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        // 屏蔽INFO级别的日志
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
} gLogger;

// 加载模型
std::vector<unsigned char> loadEngineModel(const std::string &fileName)
{
    std::ifstream file(fileName, std::ios::binary);        // 以二进制方式读取
    assert(file.is_open() && "load engine model failed!"); // 断言

    file.seekg(0, std::ios::end); // 定位到文件末尾
    size_t size = file.tellg();   // 获取文件大小

    std::vector<unsigned char> data(size); // 创建一个vector，大小为size
    file.seekg(0, std::ios::beg);          // 定位到文件开头
    file.read((char *)data.data(), size);  // 读取文件内容到data中
    file.close();

    return data;
}

int main()
{
    // ==================== 1. 创建一个runtime对象 ====================
    TRTLogger logger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(logger);

    // ==================== 2. 反序列化生成engine ====================
    // 读取文件
    auto engineModel = loadEngineModel("./model/mlp.engine");
    // 调用runtime的反序列化方法，生成engine，参数分别是：模型数据地址，模型大小，pluginFactory
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engineModel.data(), engineModel.size(), nullptr);

    if (!engine)
    {
        std::cout << "deserialize engine failed!" << std::endl;
        return -1;
    }

    // ==================== 3. 创建一个执行上下文 ====================
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    // ==================== 4. 填充数据 ====================

    // 设置stream 流
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // 数据流转：host --> device ---> inference ---> host

    // 输入数据
    float *host_input_data = new float[3]{2, 4, 8}; // host 输入数据
    int input_data_size = 3 * sizeof(float);        // 输入数据大小
    float *device_input_data = nullptr;             // device 输入数据

    // 输出数据
    float *host_output_data = new float[2]{0, 0}; // host 输出数据
    int output_data_size = 2 * sizeof(float);     // 输出数据大小
    float *device_output_data = nullptr;          // device 输出数据

    // 申请device内存
    cudaMalloc((void **)&device_input_data, input_data_size);
    cudaMalloc((void **)&device_output_data, output_data_size);

    // host --> device
    // 参数分别是：目标地址，源地址，数据大小，拷贝方向
    cudaMemcpyAsync(device_input_data, host_input_data, input_data_size, cudaMemcpyHostToDevice, stream);

    // bindings告诉Context输入输出数据的位置
    float *bindings[] = {device_input_data, device_output_data};

    // ==================== 5. 执行推理 ====================
    bool success = context -> enqueueV2((void **) bindings, stream, nullptr);
    // 数据从device --> host
    cudaMemcpyAsync(host_output_data, device_output_data, output_data_size, cudaMemcpyDeviceToHost, stream);
    // 等待流执行完毕
    cudaStreamSynchronize(stream);
    // 输出结果
    std::cout << "输出结果: " << host_output_data[0] << " " << host_output_data[1] << std::endl;

    // ==================== 6. 释放资源 ====================
    cudaStreamDestroy(stream);
    cudaFree(device_input_data); 
    cudaFree(device_output_data);

    delete host_input_data;
    delete host_output_data;

    delete context;
    delete engine;
    delete runtime;
    
    return 0;
}