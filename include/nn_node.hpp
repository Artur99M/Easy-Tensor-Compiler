#pragma once

#include "tensor.hpp"
#include <vector>
#include <memory>
#include <cstdint>
#include <tuple>
#include <stdexcept>
#include <unistd.h>

namespace etc {

//INode
class INode {

public:
             INode() = default;
    virtual ~INode() = default;

public:
    virtual        const Tensor&     evaluate    () const = 0;
    virtual inline       bool        solved      () const = 0;
    virtual inline       bool        is_operation() const    ;
    virtual inline       bool        is_number   () const    ;
    virtual              std::string dump        () const = 0;
};

//IWeight and InputData
class IWeight : private Tensor, public INode {
public:
     IWeight(const Tensor&)                   ;
     IWeight(Tensor&&)                        ;
    ~IWeight()              override = default;

public:
    const Tensor& evaluate() const override;
    inline bool   solved  () const override;
    std::string   dump    () const override;
};

class InputData : public IWeight { //I wanted make equal classes, but it is necessary to do another names
public:
    using IWeight::IWeight;
    ~InputData() override = default;
    std::string   dump    () const override;

};
// using InputData = IWeight;

class INumber : public INode {
private:
    double num_;

public:
    INumber(double);
public:
    double        val      () const;
    const Tensor& evaluate () const override;
    inline bool   solved   () const override;
    inline bool   is_number() const override;
    std::string   dump     () const override;
};

//IOperation

class IOperation : public INode {
protected:
    mutable Tensor result_{0, 0, 0, 0};
public:
     IOperation()          = default;
    ~IOperation() override = default;

public:
    virtual       void                                setArgs     (const std::vector<std::shared_ptr<INode>>& args)                = 0;
    virtual const std::vector<std::shared_ptr<INode>> getArgs     ()                                                const          = 0;
    virtual       std::string                         str_type    ()                                                const          = 0;
    inline        bool                                solved      ()                                                const override    ;
    inline        bool                                is_operation()                                                const override    ;
                  std::string                         dump        ()                                                const override    ;
};

//BinaryOperations
class BinaryOperation : public IOperation {
protected:
    std::shared_ptr<INode> lhs_;
    std::shared_ptr<INode> rhs_;
public:
     BinaryOperation(const std::shared_ptr<INode>& lhs, const std::shared_ptr<INode>& rhs)                   ;
     BinaryOperation(const Tensor& lhs,                 const std::shared_ptr<INode>& rhs)                   ;
     BinaryOperation(const std::shared_ptr<INode>& lhs, const Tensor&                 rhs)                   ;
    ~BinaryOperation()                                                                     override = default;

public:
    void                                      setArgs(const std::vector<std::shared_ptr<INode>>& args)       override;
    const std::vector<std::shared_ptr<INode>> getArgs()                                                const override;
    const Tensor& evaluate() const override = 0;
};

class ScalarAddOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
public:
    const Tensor& evaluate() const override;
    std::string   str_type() const override;
};

class ScalarSubOperation : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
public:
    const Tensor& evaluate() const override;
    std::string   str_type() const override;
};

class ScalarMulOperation : public BinaryOperation {
    double number_; // it was necessary because I don't use dynamic_cast always
public:
    //lhs_ is tensor, rhs_ is number
    ScalarMulOperation(const std::shared_ptr<INode>& tensor, double number);
    ScalarMulOperation(double number, const std::shared_ptr<INode>& tensor);
    ScalarMulOperation(const std::shared_ptr<INode>& tensor, const INumber& number);
    ScalarMulOperation(const std::shared_ptr<INode>& tensor, const std::shared_ptr<INumber>& number);
    ScalarMulOperation(const INumber& number, const std::shared_ptr<INode>& tensor);
    ScalarMulOperation(const std::shared_ptr<INumber>& number, const std::shared_ptr<INode>& tensor);
public:
    void                                      setArgs (const std::vector<std::shared_ptr<INode>>& args)       override;
    const Tensor&                             evaluate()                                                const override;
    std::string                               str_type()                                                const override;
};

class MatMulOperation    : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
public:
    const Tensor& evaluate() const override;
    std::string   str_type() const override;
};

class ConvolOperation    : public BinaryOperation {
public:
    using BinaryOperation::BinaryOperation;
public:
    const Tensor& evaluate() const override;
    std::string   str_type() const override;
};


//UnaryOperations

class UnaryOperation : public IOperation {
protected:
    std::shared_ptr<INode> arg_;
public:
     UnaryOperation(const std::shared_ptr<INode>& arg)                   ;
     UnaryOperation(const Tensor&                 arg)                   ;
    ~UnaryOperation()                                 override = default;

public:
    void                                      setArgs(const std::vector<std::shared_ptr<INode>>& args)       override;
    const std::vector<std::shared_ptr<INode>> getArgs()                                                const override;

};

class ReLUOperation      : public UnaryOperation {
public:
    using UnaryOperation::UnaryOperation;
public:
    const Tensor& evaluate() const override;
    std::string   str_type() const override;
};

class SoftmaxOperation   : public UnaryOperation {
public:
    using UnaryOperation::UnaryOperation;
public:
    const Tensor& evaluate() const override;
    std::string   str_type() const override;
};



//inline definitions

inline bool INode::is_operation() const {
    return false;
}

inline bool IOperation::is_operation() const {
    return true;
}

inline bool INode::is_number() const {
    return false;
}

inline bool INumber::is_number() const {
    return true;
}

inline bool IWeight::solved() const {
    return true;
}

inline bool INumber::solved() const {
    return true;
}

bool IOperation::solved() const {
    return result_.N() != 0;
}

} //namespace etc
