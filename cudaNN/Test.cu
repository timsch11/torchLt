#include "Test.h"
#include <iostream>



Test::Test() {
    
}

void Test::setVar(int val) {
    
}

int* Test::getVar() {
    return this->test;
}

int main() {
    Test t = Test();
    std::cout << (t.getVar() == nullptr);
}