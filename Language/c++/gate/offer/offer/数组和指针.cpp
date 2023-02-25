#include <iostream>
#include <vector>
#include <initializer_list>

using namespace std;

class Data {
public:
	int a = 1;
};
int main()
{
	cout << sizeof(decltype(1 + 1.5));
	return 0;
	
}
