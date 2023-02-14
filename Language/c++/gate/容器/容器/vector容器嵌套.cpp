#include<vector>
#include<iostream>
using namespace std;

// vector ÈÝÆ÷Ç¶Ì×
int main2() {
	vector<int> v1(5, 10);//5¸ö10
	vector<int> v2(5, 100);
	vector<int> v3(5, 1000);
	
	vector<vector<int>> v4;
	v4.push_back(v1);
	v4.push_back(v2);
	v4.push_back(v3);

	vector<vector<int>>::iterator it = v4.begin();
	for (; it != v4.end(); it++) {
		vector<int>::iterator mit = (*it).begin();
		for (; mit != (*it).end(); mit++) {
			cout << *mit << " ";
		}
		cout << endl;
	}
	return 0;
}