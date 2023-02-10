#include<iostream>
#include<string.h>
using namespace std;
class Animal1 {
public:
	int data;
};
// ”√virtual –ﬁ ŒºÃ≥–¿‡–Õ°£
class Sheep1 : virtual public Animal1 {};
class Tuo1 : virtual public Animal1 {};

class SheepTuo : public Sheep1, public Tuo1 {};

int main() {
	SheepTuo st;
	cout<<st.data;

	return 0;
}