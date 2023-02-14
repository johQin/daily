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
	//const throw()��ֹ����������֮ǰ�׳��쳣��
	// ����catch��e.what���Ჶ׽��std::exception
	{
		//��string��ת��Ϊchar*
		return this->msg.c_str();
	}
	~NewException(){}

};
int main() {
	try {
		throw NewException("ŶŶ�������쳣");
	}
	catch (exception &e) {
		cout << e.what() << endl;
	}
	return 0;
}