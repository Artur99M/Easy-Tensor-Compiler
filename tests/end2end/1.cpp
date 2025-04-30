#include <iostream>
#include <memory>
#include "neural_network.hpp"


int main() {
    etc::Tensor x = {{{{0}}}}, y = {{{{1}}}};

    etc::NeuralNetwork nn;

    const auto& input_data = std::make_shared<etc::InputData>(x);
    nn.addOp(std::make_shared<etc::ScalarAddOperation>(input_data, y));
    nn.addOp(std::make_shared<etc::ScalarMulOperation>(nn.infer_node(), 2));
    nn.addOp(std::make_shared<etc::MatMulOperation>(nn.infer_node(), y));

    std::cout << nn.infer().dump_init();
    std::cerr << nn.dump_graph();
}
