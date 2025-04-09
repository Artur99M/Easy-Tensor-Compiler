#include "tensor.hpp"
#include "debug.hpp"
#include <stdexcept>
#include <algorithm>
#include <sstream>

namespace etc {

// Tensor::Tensor(unsigned N, unsigned C, unsigned H, unsigned W) :
//     N_(N), C_(C), H_(H), W_(W) {
// }

Tensor::Tensor(unsigned C, unsigned H, unsigned W) :
    C_(C), H_(H), W_(W), channels_(C_, std::vector<std::vector<int>>(H_, std::vector<int>(W_, 0))) {

}

//InputIt is iterartor of container contains of containers contains of containers
template <typename InputIt>
Tensor::Tensor(InputIt begin, InputIt end) : channels_(begin, end) {

    C_ = channels_.size();
    if (C_ == 0 || (H_ = channels_[0].size()) == 0 || (W_ = channels_[0][0].size()) == 0)
        throw std::runtime_error("etc::Tensor::Tensor can\'t make");

}

Tensor::Tensor(std::initializer_list<std::initializer_list<std::initializer_list<int>>> init) :
    Tensor(init.size(), init.begin()->size(), init.begin()->begin()->size())
    {
    // set all values
    auto it1 = init.begin();
    for (size_t i = 0; i < C_; ++i, ++it1) {
        auto it2 = it1->begin();
        for (size_t j = 0; j < H_; ++j, ++it2) {
            channels_[i][j] = *it2;
        }
    }
}

Tensor::Tensor(Tensor&& other) :
C_(other.C_), H_(other.H_), W_(other.W_), channels_(std::move(other.channels_)) {
    other.C_ = other.H_ = other.W_ = 0;
}

Tensor& Tensor::operator=(Tensor&& rhs) {
    if (this == &rhs)
        return *this;

    channels_ = std::move(rhs.channels_);
    C_ = rhs.C_;
    H_ = rhs.H_;
    W_ = rhs.W_;
    rhs.C_ = rhs.H_ = rhs.W_ = 0;

    return *this;
}

Tensor& Tensor::operator+=(const Tensor& rhs) {
    if (C_ != rhs.C_ || H_ != rhs.H_ || W_ != rhs.W_)
        throw std::length_error{"size error in etc::Tensor::operator+="};

    for (unsigned i = 0; i < C_; ++i) {
        for (unsigned j = 0; j < H_; ++j) {
            for (unsigned k = 0; k < W_; ++k) {
                channels_[i][j][k] += rhs.channels_[i][j][k];
            }
        }
    }

    return *this;
}

Tensor Tensor::operator+(const Tensor& rhs) const {
    Tensor res = *this;
    res += rhs;
    return res;
}

Tensor& Tensor::operator-=(const Tensor& rhs) {
    if (C_ != rhs.C_ || H_ != rhs.H_ || W_ != rhs.W_)
        throw std::length_error{"size error in etc::Tensor::operator-="};

    for (unsigned i = 0; i < C_; ++i) {
        for (unsigned j = 0; j < H_; ++j) {
            for (unsigned k = 0; k < W_; ++k) {
                channels_[i][j][k] -= rhs.channels_[i][j][k];
            }
        }
    }

    return *this;
}

Tensor Tensor::operator-(const Tensor& rhs) const {
    Tensor res = *this;
    res -= rhs;
    return res;
}

Tensor operator*(const Tensor& tensor, const int number) {

    Tensor res = tensor;

    for (unsigned i = 0; i < tensor.C(); ++i)
        for (unsigned j = 0; j < tensor.H(); ++j)
            for (unsigned k = 0; k < tensor.W(); ++k)
                res[i][j][k] *= number;

    return res;
}

Tensor operator*(const int number, const Tensor& tensor) {

    Tensor res = tensor;

    for (unsigned i = 0; i < tensor.C(); ++i)
        for (unsigned j = 0; j < tensor.H(); ++j)
            for (unsigned k = 0; k < tensor.W(); ++k)
                res[i][j][k] *= number;

    return res;
}

void Tensor::set_size(unsigned C, unsigned H, unsigned W) {
    C_ = C;
    H_ = H;
    W_ = W;

    channels_.resize(C_, std::vector<std::vector<int>>(H_));
    for (size_t i = 0; i < C_; ++i) {
        channels_[i].resize(H_, std::vector<int>(W_, 0));
    }
}

Tensor& Tensor::operator*=(const Tensor& rhs) {
    return *this = *this * rhs;
}

Tensor Tensor::operator*(const Tensor& rhs) const {

    if (W_ != rhs.H_ || C_ != rhs.C_)
        throw std::logic_error("etc::Tensor::operator*(const Tensor&) can\'t mul");

    Tensor res{C_, H_, rhs.W_},
           T_rhs = rhs.transpose();

    for (unsigned c = 0; c < C_; ++c) {
        for (unsigned h = 0; h < H_; ++h) {
            for (unsigned w = 0; w < rhs.W_; ++w) {

                int sum = 0;
                for (unsigned i = 0; i < W_; ++i)
                    sum += operator[](c)[h][i] * T_rhs[c][w][i];

                res[c][h][w] = sum;
            }
        }
    }

    return res;
}

Tensor Tensor::transpose() const {

    Tensor T(C_, W_, H_);
    for (unsigned c = 0; c < C_; ++c) {

        for (unsigned i = 0; i < H_; ++i)
            for (unsigned j = 0; j < W_; ++j) {
                T[c][j][i] = operator[](c)[i][j]; // miss cache line
            }
    }

    return T;
}

std::string Tensor::dump() const {

    std::ostringstream str;
    for (unsigned c = 0; c < C_; ++c) {
        str << "channel = " << c << " {\n";
        for (unsigned h = 0; h < H_; ++h) {
            for (unsigned w = 0; w < W_; ++w)
                str << operator[](c)[h][w] << ' ';
            str << '\n';
        }
        str << "}\n";
    }

    return str.str();
}

Tensor scal_mul(const Tensor& lhs, const Tensor& rhs) {
    unsigned C = lhs.C(), H = lhs.H(), W = lhs.W();
    if (C != rhs.C() || H != rhs.H() || W != rhs.W())
        throw std::length_error("etc::scal_mul took tensors with not equal sizes");

    Tensor res{C, H, W};
    for (unsigned c = 0; c < C; ++c) {
        for (unsigned h = 0; h < H; ++h)
            for (unsigned w = 0; w < W; ++w)
                res[c][h][w] = lhs[c][h][w] * rhs[c][h][w];
    }

    return res;
}
} //namespace etc
