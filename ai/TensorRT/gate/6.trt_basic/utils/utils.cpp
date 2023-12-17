//
// Created by buntu on 2023/12/16.
//
// logger用来管控打印日志级别
// TRTLogger继承自nvinfer1::ILogger
#include "./utils.h"


// 保存权重
void utils::saveWeights(const std::string &filename, const float *data, int size)
{
    std::ofstream outfile(filename, std::ios::binary);          // #include <fstream>
    assert(outfile.is_open() && "save weights failed");  // assert断言，如果条件不满足，就会报错
    outfile.write((char *)(&size), sizeof(int));         // 保存权重的大小
    outfile.write((char *)(data), size * sizeof(float)); // 保存权重的数据
    outfile.close();
}
// 读取权重
std::vector<float> utils::loadWeights(const std::string &filename)
{
    std::ifstream infile(filename, std::ios::binary);
    assert(infile.is_open() && "load weights failed");
    int size;
    infile.read((char *)(&size), sizeof(int));                // 读取权重的大小
    std::vector<float> data(size);                            // 创建一个vector，大小为size
    infile.read((char *)(data.data()), size * sizeof(float)); // 读取权重的数据
    infile.close();
    return data;
}
int utils::getExeAbspath(char * path, int length){
    bzero(path,length);
    int ret = readlink("/proc/self/exe",path,length);   // 返回值是实际可执行文件绝对路径的长度
    if(ret == -1)
    {
        printf("----get exec abspath fail!!\n");
        return -1;
    }
    path[ret]= '\0';
    return 1;
}
int utils::getExeWd(char *wd, int wd_size){
    char path[1024]={0};
    if(getExeAbspath(path,1024) == -1){
        return -1;
    }
    char * ptr = strrchr(path, '/');
    if(ptr == NULL){
        return -1;
    }
    int a = ptr - path;
    if (wd_size<a){
        printf("---wd length is shorter than result");
        return -1;
    }
    strncpy(wd,path,a);
    return 1;
}
// 加载模型
std::vector<unsigned char> utils::loadEngineModel(const std::string &fileName)
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