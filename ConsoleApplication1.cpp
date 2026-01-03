#include <iostream>
#include <list>
#include <vector>
#include <random>
#include <string>
#include "mnist/include/mnist/mnist_reader_less.hpp"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include "definationHeader.h"
#include "realizationHeader.h"

int main()
{
    auto dataset = mnist::read_dataset<uint8_t, uint8_t>();
    std::cout << "1 - Lear Machine\n";
    int key = 0;
    std::cin >> key;
    switch (key) {
    case 1: 
        start_learning_machine(dataset);
        break;
    }
}