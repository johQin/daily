#include<iostream>
#include<exception>
#include<string.h>
using namespace std;
class NewException : public exception {
private:
	string msg;
public:
	NewException(){}
	NewException(string msg) {
		this->msg = msg;
	}
	const char* what() const throw()
	//const throw()防止父类在子类之前抛出异常，
	// 否则catch中e.what将会捕捉到std::exception
	{
		//将string类转换为char*
		return this->msg.c_str();
	}
	~NewException(){}

};
int main() {
	try {
		throw NewException("哦哦，发生异常");
	}
	catch (exception &e) {
		cout << e.what() << endl;
	}
	return 0;
}