/*
文件名使用.cu，是希望使用nvcc编译，并会自动连接cuda库

TensorRT runtime 推理流程


1.创建runtime
2.反序列plan文件，创建engine：runtime --> engine
3.创建执行上下文ExecutionContext：engine --> ExecutionContext
4.填充数据
5.执行推理：context --> enqueueV2
6.释放资源：delete
*/

#include <iostream>
#include <vector>
#include <cassert>
#include <fstream>
#include <memory>

#include <cuda_runtime.h>
#include <NvInfer.h>

// logger
class TRTLogger : public nvinfer1::ILogger
{
    void log(Severity severity, const char *msg) noexcept override
    {
        if (severity != Severity::kINFO)
            std::cout << msg << std::endl;
    }
};

// 读取engine文件
std::vector<unsigned char> loadEngineModel(const std::string &fileName)
{
    std::ifstream file(fileName, std::ios::binary);
    assert(file.is_open() && "load engine model failed");

    file.seekg(0, std::ios::end); // 移动到文件末尾
    size_t size = file.tellg();   // 计算文件大小，单位是字节

    file.seekg(0, std::ios::beg);                           // 移动到文件开头
    std::vector<unsigned char> data(size);                  // 创建一个unsigned char类型的vector，大小为size
    file.read(reinterpret_cast<char *>(data.data()), size); // 读取文件内容到data中
    return data;
}

int main()
{
    // =============== 1. 创建runtime ===============
    TRTLogger gLogger;
    nvinfer1::IRuntime *runtime = nvinfer1::createInferRuntime(gLogger);

    // ================= 2. 反序列化生成engine =================
    // 读取engine文件
    auto engineData = loadEngineModel("./model/mlp.engine");
    // 反序列化生成engine，参数：数据地址，数据大小，pluginFactory
    nvinfer1::ICudaEngine *engine = runtime->deserializeCudaEngine(engineData.data(), engineData.size(), nullptr);
    if (!engine)
    {
        std::cout << "deserializeCudaEngine failed" << std::endl;
        return -1;
    }

    // ================= 3. 创建执行上下文 =================
    nvinfer1::IExecutionContext *context = engine->createExecutionContext();

    // ================= 4. 填充数据 =================
    // 设置stream
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);

    // 输入数据
    float *h_in_data = new float[3]{2, 4, 8}; // host输入数据
    int in_data_size = sizeof(float) * 3;     // 输入数据大小
    float *d_in_data = nullptr;               // device输入数据
    // 输出数据
    float *h_out_data = new float[2]{0.0, 0.0}; // host输出数据
    int out_data_size = sizeof(float) * 2;      // 输出数据大小
    float *d_out_data = nullptr;                // device输出数据
    // 申请GPU上的内存
    cudaMalloc(&d_in_data, in_data_size);
    cudaMalloc(&d_out_data, out_data_size);
    // 异步拷贝数据到GPU，函数会立即返回，不会等待拷贝完成，但操作会被加入到stream中
    cudaMemcpyAsync(d_in_data, h_in_data, in_data_size, cudaMemcpyHostToDevice, stream);
    // binddings告诉context输入和输出的地址
    float *bindings[] = {d_in_data, d_out_data};

    // ================= 5. 执行推理 =================
    bool success = context->enqueueV2((void **)bindings, stream, nullptr);
    // 将结果拷贝回host
    cudaMemcpyAsync(h_out_data, d_out_data, out_data_size, cudaMemcpyDeviceToHost, stream);
    // stream同步，等待拷贝完成
    cudaStreamSynchronize(stream);
    // 输出
    std::cout << "输出结果: " << h_out_data[0] << " " << h_out_data[1] << std::endl;

    // ================= 6. 释放资源 =================
    cudaStreamDestroy(stream);
    cudaFree(d_in_data);
    cudaFree(d_out_data);
    delete h_in_data;
    delete h_out_data;

    delete context;
    delete engine;
    delete runtime;

    return 0;
}