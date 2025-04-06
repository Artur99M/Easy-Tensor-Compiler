#include "tensor.hpp"
#include <stdexcept>
#include <algorithm>

namespace etc {

// Tensor::Tensor(std::uint16_t N, std::uint16_t C, std::uint16_t H, std::uint16_t W) :
//     N_(N), C_(C), H_(H), W_(W) {
// }

Tensor::Tensor(std::uint16_t C, std::uint16_t H, std::uint16_t W) :
    C_(C), H_(H), W_(W), channels_(C_, std::vector<std::vector<int>>(H_)) {

    // set all values 0
    for (size_t i = 0; i < C_; ++i) {
        channels_[i].resize(H_, std::vector<int>(W_, 0));
    }

}

//InputIt is iterartor of container contains of containers contains of containers
template <typename InputIt>
Tensor::Tensor(InputIt begin, InputIt end) : channels_(begin, end) {

    C_ = channels_.size();
    if (C_ > 0) {
        H_ = channels_[0].size();
        if (H_ > 0) {
            W_ = channels_[0][0].size();
        }
    }

    // set all values 0
    for (size_t i = 0; i < C_; ++i) {
        channels_[i].resize(H_, std::vector<int>(W_, 0));
    }

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

    for (std::uint16_t i = 0; i < C_; ++i) {
        for (std::uint16_t j = 0; j < H_; ++j) {
            for (std::uint16_t k = 0; k < W_; ++k) {
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

    for (std::uint16_t i = 0; i < C_; ++i) {
        for (std::uint16_t j = 0; j < H_; ++j) {
            for (std::uint16_t k = 0; k < W_; ++k) {
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

    for (std::uint16_t i = 0; i < tensor.C(); ++i)
        for (std::uint16_t j = 0; j < tensor.H(); ++j)
            for (std::uint16_t k = 0; k < tensor.W(); ++k)
                res[i][j][k] *= number;

    return res;
}

Tensor operator*(const int number, const Tensor& tensor) {

    Tensor res = tensor;

    for (std::uint16_t i = 0; i < tensor.C(); ++i)
        for (std::uint16_t j = 0; j < tensor.H(); ++j)
            for (std::uint16_t k = 0; k < tensor.W(); ++k)
                res[i][j][k] *= number;

    return res;
}

void Tensor::set_size(std::uint16_t C, std::uint16_t H, std::uint16_t W) {
    C_ = C;
    H_ = H;
    W_ = W;

    channels_.resize(C_, std::vector<std::vector<int>>(H_));
    for (size_t i = 0; i < C_; ++i) {
        channels_[i].resize(H_, std::vector<int>(W_, 0));
    }
}

} //namespace etc
