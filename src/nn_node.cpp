#include "nn_node.hpp"
#include "tensor.hpp"
#include <algorithm>
#include <iostream>
#include <typeinfo>
#include <sstream>

namespace etc {

//INode


//IWeight

IWeight::IWeight(const Tensor& tensor) : Tensor(tensor) {
}

IWeight::IWeight(Tensor&& tensor) : Tensor(std::move(tensor)) {
}

const Tensor& IWeight::evaluate() const {
    return static_cast<const Tensor&>(*this);
}

std::string IWeight::dump() const {
    std::ostringstream str;

    str << '\"' << static_cast<const Tensor&>(*this).dump() << "\"\n";

    return str.str();
}

std::string InputData::dump() const {
    return "INPUT";
}
//INumber

INumber::INumber(double num) : num_(num) {
}

const Tensor& INumber::evaluate() const {
    throw std::logic_error("evaluate for INumber");
}

double INumber::val() const {
    return num_;
}
std::string INumber::dump() const {
    std::ostringstream str;

    str << '\"' << num_ << "\"\n";

    return str.str();
}

//IOperation

std::string IOperation::dump() const {
    std::ostringstream str;

    for (const std::shared_ptr<INode>& i : getArgs()) {
        str << i->dump() <<  " -> \"" << str_type() << '\n' << this << "\"\n";
    }

    str << '\"' << str_type() << '\n' << this << "\"\n";

    return str.str();
}

//BinaryOperation

BinaryOperation::BinaryOperation(const std::shared_ptr<INode>& lhs, const std::shared_ptr<INode>& rhs) :
    lhs_(lhs), rhs_(rhs) {
}

BinaryOperation::BinaryOperation(const Tensor& lhs, const std::shared_ptr<INode>& rhs):
    lhs_(std::make_shared<IWeight>(lhs)), rhs_(rhs) {
}

BinaryOperation::BinaryOperation(const std::shared_ptr<INode>& lhs, const Tensor& rhs) :
    lhs_(lhs), rhs_(std::make_shared<IWeight>(rhs)) {
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

// BinaryOperations

const Tensor& ScalarAddOperation::evaluate() const {
    if (result_.N() == 0)
        result_ = lhs_->evaluate() + rhs_->evaluate();

    return result_;
}
std::string ScalarAddOperation::str_type() const {
    return "ScalarAddOperation";
}

const Tensor& ScalarSubOperation::evaluate() const {
    if (result_.N() == 0)
        result_ = lhs_->evaluate() - rhs_->evaluate();

    return result_;
}

std::string ScalarSubOperation::str_type() const {
    return "ScalarSubOperation";
}

const Tensor& MatMulOperation::evaluate() const {
    if (result_.N() == 0)
        result_ = lhs_->evaluate() * rhs_->evaluate();

    return result_;
}

std::string MatMulOperation::str_type() const {
    return "MatMulOperation";
}

const Tensor& ConvolOperation::evaluate() const {
    if (result_.N() == 0)
        result_ = convol(lhs_->evaluate(), rhs_->evaluate());

    return result_;
}

std::string ConvolOperation::str_type() const {
    return "ConvolOperation";
}

// ScalarMulOperation

ScalarMulOperation::ScalarMulOperation(const std::shared_ptr<INode>& tensor, double number) :
    BinaryOperation(tensor, std::make_shared<INumber>(number)), number_(number) {
}

ScalarMulOperation::ScalarMulOperation(double number, const std::shared_ptr<INode>& tensor) :
    BinaryOperation(tensor, std::make_shared<INumber>(number)), number_(number) {
}

ScalarMulOperation::ScalarMulOperation(const std::shared_ptr<INode>& tensor, const INumber& number) :
    BinaryOperation(tensor, std::make_shared<INumber>(number)), number_(number.val()) {
}

ScalarMulOperation::ScalarMulOperation(const INumber& number, const std::shared_ptr<INode>& tensor) :
    BinaryOperation(tensor, std::make_shared<INumber>(number)), number_(number.val()) {
}

ScalarMulOperation::ScalarMulOperation(const std::shared_ptr<INumber>& number, const std::shared_ptr<INode>& tensor) :
    BinaryOperation(tensor, number), number_(number->val()){
}

ScalarMulOperation::ScalarMulOperation(const std::shared_ptr<INode>& tensor, const std::shared_ptr<INumber>& number) :
    BinaryOperation(tensor, number), number_(number->val()){
}

const Tensor& ScalarMulOperation::evaluate() const {
    if (result_.N() == 0)
        result_ = lhs_->evaluate() * number_;

    return result_;
}

std::string ScalarMulOperation::str_type() const {
    return "ScalarMulOperation";
}

void ScalarMulOperation::setArgs(const std::vector<std::shared_ptr<INode>>& args) {
    if (args.size() != 2)
        throw std::range_error("etc::BinaryOperation::setArgs args.size() != 2");

    if (typeid(*(args[0])) == typeid(INumber)) {
        number_ = dynamic_cast<const INumber&>(*(args[0])).val();
        rhs_    = args[0];
        lhs_    = args[1];

    } else if (typeid(*(args[1])) == typeid(INumber)) {
        number_ = dynamic_cast<const INumber&>(*(args[0])).val();
        lhs_    = args[0];
        rhs_    = args[1];

    } else
        throw std::runtime_error("No number for etc::ScalarMulOperation::setArgs");
}

// UnaryOperation

UnaryOperation::UnaryOperation(const std::shared_ptr<INode>& arg) : arg_(arg) {
}

UnaryOperation::UnaryOperation(const Tensor& arg) : arg_(std::make_shared<IWeight>(arg)) {
}
void UnaryOperation::setArgs(const std::vector<std::shared_ptr<INode>>& args) {
    if (args.size() != 1)
        throw std::range_error("etc::UnaryOperation::setArgs args.size() != 1");

    arg_ = args[0];
}

const std::vector<std::shared_ptr<INode>> UnaryOperation::getArgs() const {
    return std::vector<std::shared_ptr<INode>>({arg_});
}

const Tensor& ReLUOperation::evaluate() const {
    if (result_.N() == 0)
        result_ = arg_->evaluate().ReLU();

    return result_;
}

std::string ReLUOperation::str_type() const {
    return "ReLUOperation";
}

const Tensor& SoftmaxOperation::evaluate() const {
    if (result_.N() == 0)
        result_ = arg_->evaluate().softmax();

    return result_;
}
std::string SoftmaxOperation::str_type() const {
    return "SoftmaxOperation";
}

} //namespace etc
