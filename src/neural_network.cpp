#include "neural_network.hpp"
#include <unordered_set>
#include <stack>


namespace etc {

std::shared_ptr<IOperation> NeuralNetwork::addOp(std::shared_ptr<IOperation> op) {
    return op;
}

void NeuralNetwork::set_infer(const std::shared_ptr<INode>& infer) {
    infer_ = infer;
}

Tensor NeuralNetwork::infer() {
    std::stack<std::shared_ptr<INode>> stk;

    // reducing recursion to a loop
    std::unordered_set<size_t> infers;
    std::stack<std::shared_ptr<INode>> stk_bypass;

    stk_bypass.push(root_);
    infers.emplace(static_cast<size_t>(root_.get());

    INode* cur = root_;
    while (!stk_bypass.empty()) {
        bool find = false;
        for (INode* x : stk_bypass.top()->children()) {
            if (!infers.find(x)) {
                infers.emplace(static_cast<size_t>(x));
                stk_bypass.push(cur);
                cur = x;
                find = true;
            }
        }

        if (!find) {
            stk.push(cur);
            if (stk_bypass.size() == 0)
                break;
            cur = stk_bypass.top().get();
            stk_bypass.pop();
        }
    }

    while (!stk.empty()) {
        stk.top()->evaluate();
        stk.pop();
    }

    return infer_->evaluate();
}

} //namespace etc
