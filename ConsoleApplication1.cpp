#include <iostream>
#include "mnist/include/mnist/mnist_reader_less.hpp"
#include "definationHeader.h"
#include "realizationHeader.h"

int main()
{
    auto dataset = mnist::read_dataset<uint8_t, uint8_t>();
    std::cout << "1 - Lear Machine\n";
    std::cout << "2 - Start Working\n";
    int key = 0;
    std::cin >> key;
    switch (key) {
    case 1: 
        
        start_learning_machine(dataset);
        break;
    case 2: break;
    }
}