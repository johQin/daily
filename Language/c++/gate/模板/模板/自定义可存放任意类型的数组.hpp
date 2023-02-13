#pragma once
#include<iostream>
#include<string.h>
using namespace std;
template<class T>
class MyArray {
private:
	T* arr;
	int length;
	int capacity;
public:
	MyArray();
	MyArray(int c);
	MyArray(const MyArray& ob);
	~MyArray();
	MyArray& operator=(MyArray& ob);


	void pushBack(T elem);
	void toString();
};

template<class T>
inline MyArray<T>::MyArray()
{
	capacity = 5;
	length = 0;
	arr = new T[capacity];
	memset(arr, 0, sizeof(T) * capacity);
	// memset是计算机中C / C++语言初始化函数。
	// 作用是将某一块内存中的内容全部设置为指定的值， 这个函数通常为新申请的内存做初始化工作。
	// void* memset(void* s, int ch, size_t n);
	// 函数解释：将s中当前位置后面的n个字节 （typedef unsigned int size_t ）用 ch 替换并返回 s 。
}

template<class T>
inline MyArray<T>::MyArray(int c)
{
	capacity = c;
	length = 0;
	arr = new T[capacity];
	memset(arr, 0, sizeof(T) * capacity);
}

template<class T>
inline MyArray<T>::MyArray(const MyArray& ob)
{
	capacity = ob.capacity;
	length = ob.length;
	arr = new T[capacity];
	memset(arr, 0, sizeof(T) * capacity);
	memcpy(arr, ob.arr, sizeof(T) * capacity);
}

template<class T>
inline MyArray<T>::~MyArray()
{
	delete[] arr;
}

template<class T>
inline MyArray<T>& MyArray<T>::operator=(MyArray& ob)
{
	if (arr != NULL) {
		delete[] arr;
		arr = NULL;
	}
	length = ob.length;
	capacity = ob.capacity;
	arr = new T[capacity];
	memset(arr, 0, sizeof(T) * capacity);
	memcpy(arr, ob.arr, sizeof(T) * capacity);
	return *this;
}

template<class T>
inline void MyArray<T>::pushBack(T elem)
{
	if (length == capacity) {
		capacity = 2 * capacity;
		T* tmp = new T[capacity];
		if (arr != NULL) {
			memcpy(tmp, arr, sizeof(T) * length);
			delete[] arr;
		}
		arr = tmp;
	}
	arr[length] = elem;
	length++;
}

template<class T>
inline void MyArray<T>::toString()
{
	for (int i = 0; i < length; i++) {
		cout << arr[i]<<" ";
	}
	cout << endl;
}

