#include<stdio.h>
#include<iostream>
#include<assert.h>
using namespace std;

int main() {
    int* p = NULL; //初始化置NULL

    p = (int*)malloc(sizeof(int) * 3); //申请n个int内存空间
    assert(p != NULL); //判空，防错设计
}