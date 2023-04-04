#include<stdio.h>
void insertion_sort(int arr[], int len) {
	int i, j, key;
	for (i = 1; i < len; i++) {
		key = arr[i];
		j = i - 1;
		while (j >= 0 && arr[j] > key) {
			arr[j+1] = arr[j];
			j--;
		}
		arr[j + 1] = key;
	}
}
int main() {
	int arr[] = { 10,15,9,25,47,12,17,12,33,31 };
	int len = (int)sizeof(arr) / sizeof(*arr);
	insertion_sort(arr, len);
	for (int i = 0; i < len; i++) {
		printf("%d\t", arr[i]);
	}
	return 0;
}