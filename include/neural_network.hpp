#pragma once

#include "nn_node.hpp"

namespace etc {

class NeuralNetwork {
    std::shared_ptr<INode> root_ {nullptr};
    std::shared_ptr<INode> infer_{nullptr};
public:
    std::shared_ptr<IOperation> addOp(std::shared_ptr<IOperation> op);
    Tensor infer();
    void set_infer(const std::shared_ptr<INode>& infer_);
};
} //namespace etc
