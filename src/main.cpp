#include "nn_node.hpp"
#include "tensor.hpp"
#include <iostream>
#include <algorithm>
#include <ctime>

using namespace etc;

int main() {
    Tensor x = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    Tensor z = {
               {
               {
               {-4, -3},
               {2, 1}
               },
               {
               {-8, -7},
               {6, 5}
               }
               },
               {
               {
               {-12, -11},
               {10, 9}
               },
               {
               {-16, -15},
               {14, 13}
               }
               }
               };

    std::cout << convol(x, z).dump();
}
