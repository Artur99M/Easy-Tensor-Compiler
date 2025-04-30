#include "neural_network.hpp"
#include <deque>
#include <memory>
#include <sstream>
#include <typeinfo>
#include <iostream>
#include <stack>
#include <tuple>
#include "debug.hpp"

namespace etc {

const std::shared_ptr<IOperation>& NeuralNetwork::addOp(const std::shared_ptr<IOperation>& op) {

    if (infer_ == nullptr) {
        infer_ = op;
        return op;
    }

    for (auto& i : op->getArgs())
        if (i == infer_) {
            infer_ = op;
            break;
        }

    return op;
}

const Tensor& NeuralNetwork::infer() {

    if (!infer_->solved()) {

        //here I tried to avoid recursive call
        // if true: node was visited
        std::stack<std::pair<IOperation*, bool>> stk;
        if (infer_->is_operation())
            stk.push(std::make_pair(reinterpret_cast<IOperation*>(infer_.get()), false));

        //Post Order if we consider infer_ is root
        while (!stk.empty()) {
            std::pair<IOperation*, bool>& top = stk.top();

            if (top.second == true) { // if node was visited but maybe hasn't be solved
                top.first->evaluate();
                stk.pop();
            } else { // node wasn't visited => it's clildren wasn't visited too
                top.second = true;
                std::vector<std::shared_ptr<INode>> args = top.first->getArgs();

                for (auto& i : args)
                    if (i->is_operation())
                        stk.push(std::make_pair(reinterpret_cast<IOperation*>(i.get()), false));
            }

        }
    }

    return infer_->evaluate();
}

std::string NeuralNetwork::dump_graph() const {

    std::ostringstream str;
    str << "digraph G {\n" << infer_->dump() << " -> OUTPUT\n}";
    return str.str();
}


std::shared_ptr<INode>& NeuralNetwork::infer_node() {
    return infer_;
}

const std::shared_ptr<INode>& NeuralNetwork::infer_node() const {
    return infer_;
}
} //namespace etc
