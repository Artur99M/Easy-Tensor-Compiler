#include "tensor.hpp"
#include "debug.hpp"
#include <stdexcept>
#include <algorithm>
#include <sstream>
#include <cmath>

namespace etc {

//      ___________    ________________     ___________        ___________
//    //           \\  ________________   //           \\    //           \\
//   //                       ||         //             \\  ||             ||
//  ||                        ||        ||               || ||             ||
//  ||                        ||        ||               || ||____________//
//  ||                        ||        ||               || ||        \\
//  ||                        ||        ||               || ||          \\
//   \\                       ||         \\             //  ||           \\
//    \\___________//         ||          \\___________//   ||            \\


Tensor::Tensor(unsigned N, unsigned C, unsigned H, unsigned W) :
    N_(N), C_(C), H_(H), W_(W), data_(N_ * C_ * H_ * W_) {
}

Tensor::Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<double>>>> init) :
    Tensor(init.size(), init.begin()->size(), init.begin()->begin()->size(), init.begin()->begin()->begin()->size())
    {
    // set all values
    auto it1 = init.begin();
     for (unsigned b = 0; b < N_; ++b, ++it1) {
        auto it2 = it1->begin();
        for (unsigned c = 0; c < C_; ++c, ++it2) {
            auto it3 = it2->begin();
            for (unsigned h = 0; h < H_; ++h, ++it3) {
                auto it4 = it3->begin();
                for (unsigned w = 0; w < W_; ++w, ++it4) {
                    data_[w + h * W_ + c * H_ * W_ + b * C_ * W_ * H_] = *it4;
                }
            }
        }
     }
}

Tensor::Tensor(Tensor && other) :
N_(other.N_), C_(other.C_), H_(other.H_), W_(other.W_), data_(std::move(other.data_)) {
    other.N_ = other.C_ = other.H_ = other.W_ = 0;
    other.data_.resize(0);
}

Tensor& Tensor::operator=(Tensor&& rhs) {
    if (this == &rhs)
        return *this;

    data_ = std::move(rhs.data_);
    N_ = rhs.N_;
    C_ = rhs.C_;
    H_ = rhs.H_;
    W_ = rhs.W_;
    rhs.C_ = rhs.H_ = rhs.W_ = 0;
    rhs.data_.resize(0);

    return *this;
}


//    _           _         _______     ________________
//  // \\       // \\     //       \\   ________________ ||             ||
// ||   \\     //   ||   //         \\         ||        ||             ||
// ||    \\   //    ||  //           \\        ||        ||             ||
// ||     \\_//     || ||             ||       ||        ||             ||
// ||               || ||_____________||       ||        ||_____________||
// ||               || ||             ||       ||        ||             ||
// ||               || ||             ||       ||        ||             ||
// ||               || ||             ||       ||        ||             ||


Tensor& Tensor::operator+=(const Tensor& rhs) {
    if (N_ != rhs.N_ || C_ != rhs.C_ || H_ != rhs.H_ || W_ != rhs.W_)
        throw std::length_error{"size error in etc::Tensor::operator+="};

    for (size_t i = 0, sz = N_ * C_ * H_ * W_; i < sz; ++i)
        data_[i] += rhs.data_[i];

    return *this;
}

Tensor Tensor::operator+(const Tensor& rhs) const {
    Tensor res = *this;
    res += rhs;
    return res;
}

Tensor& Tensor::operator-=(const Tensor& rhs) {
    if (N_ != rhs.N_ || C_ != rhs.C_ || H_ != rhs.H_ || W_ != rhs.W_)
        throw std::length_error{"size error in etc::Tensor::operator+="};

    for (size_t i = 0, sz = N_ * C_ * H_ * W_; i < sz; ++i)
        data_[i] -= rhs.data_[i];

    return *this;
}

Tensor Tensor::operator-(const Tensor& rhs) const {
    Tensor res = *this;
    res -= rhs;
    return res;
}

Tensor& Tensor::operator*=(const double number) {

    for (size_t i = 0, sz = N_ * C_ * H_ * W_; i < sz; ++i)
        data_[i] *= number;

    return *this;

}

Tensor operator*(const Tensor& tensor, const double number) {
    Tensor res = tensor;
    res *= number;
    return res;
}

Tensor operator*(const double number, const Tensor& tensor) {
    Tensor res = tensor;
    res *= number;
    return res;
}

Tensor& Tensor::operator*=(const Tensor& rhs) {
    return *this = *this * rhs;
}

Tensor Tensor::operator*(const Tensor& rhs) const {

    if (W_ != rhs.H_ || C_ != rhs.C_ || N_ != rhs.N_)
        throw std::logic_error("etc::Tensor::operator*(const Tensor&) can\'t mul");

    Tensor res{N_, C_, H_, rhs.W_},
           T_rhs = rhs.transpose();

    for (unsigned b = 0; b < N_; ++b) {
        for (unsigned c = 0; c < C_; ++c) {
            for (unsigned h = 0; h < H_; ++h) {
                for (unsigned w = 0; w < rhs.W_; ++w) {

                    double sum = 0;
                    for (unsigned i = 0; i < W_; ++i)
                        sum += operator[](b)[c][h][i] * T_rhs[b][c][w][i];

                    res[b][c][h][w] = sum;
                }
            }
        }
    }

    return res;
}

Tensor Tensor::transpose() const {

    Tensor T(N_, C_, W_, H_);
    for (unsigned b = 0; b < N_; ++b)
        for (unsigned c = 0; c < C_; ++c) {

            for (unsigned i = 0; i < H_; ++i)
                for (unsigned j = 0; j < W_; ++j) {
                    T[b][c][j][i] = operator[](b)[c][i][j]; // miss cache line
                }
        }

    return T;
}

Tensor convol(const Tensor& lhs, const Tensor& rhs) {

    unsigned N = lhs.N(), C = lhs.C(),
             Hl = lhs.H(), Wl = lhs.W(),
             Hr = rhs.H(), Wr = rhs.W();

    if(C != rhs.C() || N != rhs.N() || Hl < Hr || Wl < Wr)
        throw std::length_error("etc::convol can\'t be solved");

    Tensor res {N, C, Hl - Hr + 1, Wl - Wr + 1};

    for (unsigned b = 0; b < N; ++b)
        for (unsigned c = 0; c < C; ++c)
            for (unsigned i = 0; i < Hl - Hr + 1; ++i)
                for (unsigned j = 0; j < Wl - Wr + 1; ++j) {

                    double sum = 0;

                    for (unsigned h = 0; h < Hr; ++h)
                        for (unsigned w = 0; w < Wr; ++w)
                            sum += lhs[b][c][h + i][w + j] * rhs[b][c][h][w];

                    res[b][c][i][j] = sum;
                }

    return res;

}

Tensor& Tensor::ReLU_self() {

    for (double& x : data_)
        x = (x > 0 ? x : 0);

    return *this;
}

Tensor Tensor::ReLU() const {

    Tensor res{*this};
    return res.ReLU_self();
}

// Tensor& Tensor::softmax_self() {
//
//     for (unsigned h = 0; h < H_; ++h)
//         for (unsigned w = 0; w < W_; ++w)
//         for (unsigned c = 0; c < C_; ++c) {
//             int sum = 0, sum_2 = 0;
//             for (unsigned b = 0; b < N_; ++b) {
//                 int x  = operator[](b)[c][h][w];
//                 sum   += x;
//                 sum_2 += x * x;
//             }
//             int E = sum / N_, E_2 = sum_2 / E, D = E_2 - E * E;
//             if (D == 0)
//                 throw std::runtime_error("etc::Tensor::softmax_self D == 0");
//             for (unsigned b = 0; b < N_; ++b)
//                 operator[](b)[c][h][w] = (operator[](b)[c][h][w] - E) / std::sqrt(D);
//         }
//
// }
//
// Tensor Tensor::softmax() const {
//
//     Tensor res{*this};
//     return res.softmax_self();
// }

std::string Tensor::dump() const {

    std::ostringstream str;
    for (unsigned b = 0; b < N_; ++b) {
        str << "batch = " << b << " {\n";
        for (unsigned c = 0; c < C_; ++c) {
            str << "channel = " << c << " {\n";
            for (unsigned h = 0; h < H_; ++h) {
                for (unsigned w = 0; w < W_; ++w)
                    str << operator[](b)[c][h][w] << ' ';
                str << '\n';
            }
            str << "}\n";
        }
        str << "}\n";
    }

    return str.str();
}


void Tensor::set_size(unsigned N, unsigned C, unsigned H, unsigned W) {
    N_ = N;
    C_ = C;
    H_ = H;
    W_ = W;

    data_.resize(N * C * H * W);
}

} //namespace etc
