#include<iostream>
#include "ͷԴ����.h"
// ����ڵ����ļ��У�ֻinclude .h�ļ�
// ģ��������У���Ҫ���α��룬
// ��һ������Ԥ����׶Σ�����ģ����������б���
// �ڶ������ڵ���ʵ����ʱ����ʱ��Ҫ����ģ����б��룬����ʱ��.h���Ҳ�������ʵ�֣������޷��ٴν�ʵ��ģ�����ͱ�����ʵ����
// ��������ͻ�������⣬ֱ�ӱ���undefined reference to Data<int,int>::Data()
// �������ﻹ��Ҫ��.h��ʵ��data.cpp����������
#include "ͷԴ����_��ʵ��.cpp"
using namespace std;
int main7() {
	Data<int, int> ob(10, 20);
	ob.toString();
	return 0;
}