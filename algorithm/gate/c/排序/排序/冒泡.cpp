#include<stdio.h>

void bubble_sort(int arr[], int len) {
	int tmp = 0, i, j = 0;
	for (i = 0; i < len-1; i++) {
		for (j = 0; j < len - 1 - i; j++) {
			if (arr[j] > arr[j+1]) {
				tmp = arr[j];
				arr[j] = arr[j+1];
				arr[j+1] = tmp;
			}

		}
	}
}
int main1() {
	int arr[] = { 10,15,9,25,47,12,17,12,33,31,25 };
	int len = (int) sizeof(arr) / sizeof(*arr);
	bubble_sort(arr, len);
	for (int i = 0; i < len; i++) {
		printf("%d\t", arr[i]);
	}
	return 0;
}