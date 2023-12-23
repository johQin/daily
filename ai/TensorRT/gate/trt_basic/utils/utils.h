//
// Created by buntu on 2023/12/16.
//

#ifndef TRT_DEMO_UTILS_H
#define TRT_DEMO_UTILS_H
#include <fstream>
#include <cassert>
#include <iostream>
#include <vector>
#include <string.h>
#include <unistd.h>
namespace utils{

    // 保存权重
    void saveWeights(const std::string &filename, const float *data, int size);
    // 读取权重
    std::vector<float> loadWeights(const std::string &filename);
    std::vector<unsigned char> loadEngineModel(const std::string &fileName);
    int getExeAbspath(char * path, int length);
    int getExeWd(char *wd, int wd_size);

}
#endif //TRT_DEMO_UTILS_H
