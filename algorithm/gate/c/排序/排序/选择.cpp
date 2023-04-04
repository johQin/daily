#include<stdio.h>

void swap(int* a, int* b) {
	int tmp = *a;
	*a = *b;
	*b = tmp;
}
void selection_sort(int arr[], int len) {
	int max , i, j;
	for (i = 0; i < len; i++) {
		max = len-1-i;
		for (j = 0; j < len - i; j++) {
			if (arr[j] > arr[max]) {
				max = j;
			}
		}
		swap(&arr[max], &arr[len - 1- i]);
	}
}
int main2() {
	int arr[] = { 10,15,9,25,47,12,17,12,33,31 };
	int len = (int)sizeof(arr) / sizeof(*arr);
	selection_sort(arr, len);
	for (int i = 0; i < len; i++) {
		printf("%d\t", arr[i]);
	}
	return 0;
}