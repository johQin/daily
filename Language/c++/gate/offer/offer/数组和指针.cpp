#include<stdio.h>
#include<iostream>
#include<assert.h>
using namespace std;

int main() {
    int* p = NULL; //��ʼ����NULL

    p = (int*)malloc(sizeof(int) * 3); //����n��int�ڴ�ռ�
    assert(p != NULL); //�пգ��������
}