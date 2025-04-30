#include <iostream>
#include <memory>
#include "neural_network.hpp"


int main() {
    etc::Tensor input = {
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

    etc::Tensor weight = {
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
    etc::Tensor weight2 = {
                    {{{1}}, {{1}}},
                    {{{1}}, {{1}}}
                    };

    etc::NeuralNetwork nn;

    const auto& input_data = std::make_shared<etc::InputData>(input);
    nn.addOp(std::make_shared<etc::ConvolOperation>(input_data, weight));
    nn.addOp(std::make_shared<etc::ScalarAddOperation>(nn.infer_node(), weight2));
    nn.addOp(std::make_shared<etc::ScalarMulOperation>(nn.infer_node(), 12));

    std::cout << nn.infer().dump_init();
    std::cerr << nn.dump_graph();
}
