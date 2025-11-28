#include <iostream>
int main() {
#ifdef _GLIBCXX_USE_CXX11_ABI
    std::cout << _GLIBCXX_USE_CXX11_ABI << std::endl;
#else
    std::cout << "Not defined" << std::endl;
#endif
    return 0;
}
