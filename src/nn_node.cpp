#include "nn_node.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <iostream>

namespace etc {

//INode

INode::INode(const std::vector<INode*>& children) : children_(children){
}

INode::INode(std::vector<INode*>&& children) : children_(std::move(children)){
}

void INode::add_child(INode* new_child) {
    if (new_child != nullptr)
        children_.push_back(new_child);
}

void INode::add_child(std::vector<INode*> new_children) {
    children_.insert(children_.end(), new_children.begin(), new_children.end());
}

const std::vector<INode*>& INode::children() const {
    return children_;
}


//IWeight

IWeight::IWeight(const Tensor& tensor) : Tensor(tensor) {
}

IWeight::IWeight(Tensor&& tensor) : Tensor(std::move(tensor)) {
}

Tensor IWeight::evaluate() const {
    return static_cast<Tensor>(*this);
}

//InputData

InputData::InputData(const Tensor& tensor) : val_(tensor) {
}

Tensor InputData::evaluate() const {
    return val_;
}

//BinaryOperation

BinaryOperation::BinaryOperation(const std::shared_ptr<INode>& lhs, const std::shared_ptr<INode>& rhs) :
    lhs_(lhs), rhs_(rhs) {
    lhs_->add_child(this);
    rhs_->add_child(this);
}
void BinaryOperation::setArgs(const std::vector<std::shared_ptr<INode>>& args) {
    if (args.size() != 2)
        throw std::range_error("etc::BinaryOperation::setArgs args.size() != 2");

    lhs_ = args[0];
    rhs_ = args[1];
}

const std::vector<std::shared_ptr<INode>> BinaryOperation::getArgs() const {
    return std::vector<std::shared_ptr<INode>>({lhs_, rhs_});
}

// evaluate BinaryOperation

Tensor ScalarAddOperation::evaluate() const {
    if (result_.C() == 0)
        result_ = lhs_->evaluate() + rhs_->evaluate();

    return result_;
}

Tensor ScalarSubOperation::evaluate() const {
    if (result_.C() == 0)
        result_ = lhs_->evaluate() - rhs_->evaluate();

    return result_;
}

Tensor MatMulOperation::evaluate() const {
    if (result_.C() == 0)
        result_ = lhs_->evaluate() * rhs_->evaluate();

    return result_;
}

Tensor ConvolOperation::evaluate() const {
    if (result_.C() == 0)
        result_ = convol(lhs_->evaluate(), rhs_->evaluate());

    return result_;
}

// ScalarMulOperation

ScalarMulOperation::ScalarMulOperation(const std::shared_ptr<INode>& tensor, int number) :
    BinaryOperation(tensor, nullptr), number_(number) {
}

ScalarMulOperation::ScalarMulOperation(int number, const std::shared_ptr<INode>& tensor) :
    BinaryOperation(tensor, nullptr), number_(number) {
}

Tensor ScalarMulOperation::evaluate() const {
    if (result_.C() == 0)
        result_ = lhs_->evaluate() * number_;

    return result_;
}

// UnaryOperation

UnaryOperation::UnaryOperation(const std::shared_ptr<INode> arg) : arg_(arg) {
    arg_->add_child(this);
}
void UnaryOperation::setArgs(const std::vector<std::shared_ptr<INode>>& args) {
    if (args.size() != 1)
        throw std::range_error("etc::UnaryOperation::setArgs args.size() != 1");

    arg_ = args[0];
}

const std::vector<std::shared_ptr<INode>> UnaryOperation::getArgs() const {
    return std::vector<std::shared_ptr<INode>>({arg_});
}

Tensor ReLUOperation::evaluate() const {
    if (result_.C() == 0)
        result_ = arg_->evaluate().ReLU();

    return result_;
}

} //namespace etc
