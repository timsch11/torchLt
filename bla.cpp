#include <iostream>

int main() {
    int a = 2;

    int* b = &a;

    a = 3;

    std::cout << a << "  " << *b << std::endl;
    return 0;
}