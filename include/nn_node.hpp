#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <cstdint>
#include <tuple>
#include <stdexcept>
#include <unistd.h>

namespace etc {

class INode {

private:
    std::vector<INode*> children_;

public:
    INode() = default;
    INode(const INode&) = default;
    INode(const std::vector<INode*>&);
    INode(std::vector<INode*>&&);
    const std::vector<INode*>& children() const;
    virtual ~INode() = default;

public:
    virtual Tensor evaluate() const = 0; //returns result of operation or tensor of weights
    void add_child(INode*);
    void add_child(std::vector<INode*>);
};

class IWeight final : private Tensor, public INode {
public:
    IWeight(const Tensor&);
    IWeight(Tensor&&);
    ~IWeight() override = default;

public:
    Tensor evaluate() const override;
};

class IOperation : public INode {
protected:
    mutable Tensor result_{0, 0, 0, 0};
public:
    virtual void setArgs(const std::vector<std::shared_ptr<INode>>& args) = 0;
    virtual const std::vector<std::shared_ptr<INode>> getArgs() const = 0;
};


class BinaryOperation : public IOperation {
protected:
    std::shared_ptr<INode> lhs_;
    std::shared_ptr<INode> rhs_;
public:
    BinaryOperation(const std::shared_ptr<INode>& lhs, const std::shared_ptr<INode>& rhs);
    void setArgs(const std::vector<std::shared_ptr<INode>>& args) override;
    const std::vector<std::shared_ptr<INode>> getArgs() const override;
    Tensor evaluate() const override = 0;
    ~BinaryOperation() override = default;
};

class ScalarAddOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    using BinaryOperation::getArgs;
    using BinaryOperation::setArgs;
    Tensor evaluate() const override;
};
class ScalarSubOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    using BinaryOperation::getArgs;
    using BinaryOperation::setArgs;
    Tensor evaluate() const override;
};
class ScalarMulOperation : public BinaryOperation {
    int number_;
public:
    //only lhs_ will be defined
    ScalarMulOperation(const std::shared_ptr<INode>& tensor, int number);
    ScalarMulOperation(int number, const std::shared_ptr<INode>& tensor);
    Tensor evaluate() const override;
};
class MatMulOperation    : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    using BinaryOperation::getArgs;
    using BinaryOperation::setArgs;
    Tensor evaluate() const override;
};
class ConvolOperation    : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
    using BinaryOperation::getArgs;
    using BinaryOperation::setArgs;
    Tensor evaluate() const override;
};

class UnaryOperation : public IOperation {
protected:
    std::shared_ptr<INode> arg_;
public:
    UnaryOperation(const std::shared_ptr<INode> arg);
    void setArgs(const std::vector<std::shared_ptr<INode>>& args) override;
    const std::vector<std::shared_ptr<INode>> getArgs() const override;
    Tensor evaluate() const override = 0;
    ~UnaryOperation() override = default;

};

class ReLUOperation      : public UnaryOperation {
public:
    using UnaryOperation::UnaryOperation;
    Tensor evaluate() const override;
};

class InputData : public INode {
    Tensor val_;
public:
    InputData(const Tensor& tensor);
    Tensor evaluate() const override;
};

// class SoftmaxOperation   : public UnaryOperation {
// public:
//     using UnaryOperation::UnaryOperation;
//     Tensor evaluate() const override;
// };

} //namespace etc
