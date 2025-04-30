#pragma once

#include "nn_node.hpp"

namespace etc {

class NeuralNetwork {
    std::shared_ptr<INode> infer_ = nullptr;
public:
    const std::shared_ptr<IOperation>& addOp     (const std::shared_ptr<IOperation>& op)      ;
    const Tensor&                      infer     ()                                           ;
    std::string                        dump_graph()                                      const;
    const std::shared_ptr<INode>&      infer_node()                                      const;
          std::shared_ptr<INode>&      infer_node()                                           ;
};
} //namespace etc
