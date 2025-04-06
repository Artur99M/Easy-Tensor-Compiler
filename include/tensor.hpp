#pragma once
#include <vector>
#include <stdexcept>
#include <cstdint>

namespace etc {

class Tensor {

private:
    // std::uint16_t N_ = 1;
    std::uint16_t C_ = 0,
                  H_ = 0,
                  W_ = 0;
    std::vector<std::vector<std::vector<int>>> channels_;

public:
    Tensor() = default;
    Tensor(std::uint16_t C, std::uint16_t H, std::uint16_t W);

    //InputIt is iterartor of container contains of containers contain of containers contain types casts to int
    template <typename InputIt>
    Tensor(InputIt begin, InputIt end);
    Tensor(std::initializer_list<std::initializer_list<std::initializer_list<int>>> init);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other);
    Tensor& operator=(const Tensor& rhs) = default;
    Tensor& operator=(Tensor&& rhs);
    ~Tensor() = default;

//element handing
public:
    inline      std::vector<std::vector<int>>&  operator[](std::uint16_t i) {
        return channels_[i];
    }
    inline const std::vector<std::vector<int>>& operator[](std::uint16_t i) const {
        return channels_[i];
    }
    inline      int&                            at        (std::uint16_t i, std::uint16_t j, std::uint16_t k) {
        if (i >= C_ || j >= H_ || k >= W_)
            throw std::out_of_range("in etc::Tensor::at");
        return channels_[i][j][k];
    }
    inline const int&                           at        (std::uint16_t i, std::uint16_t j, std::uint16_t k) const {
        if (i >= C_ || j >= H_ || k >= W_)
            throw std::out_of_range("in etc::Tensor::at");
        return channels_[i][j][k];
    }

//size
public:
    inline std::uint16_t C() const {
        return C_;
    }
    inline std::uint16_t H() const {
        return H_;
    }
    inline std::uint16_t W() const {
        return W_;
    }

//Math operations
public:
    Tensor& operator+=(const Tensor& rhs);
    Tensor  operator+ (const Tensor& rhs) const;
    Tensor& operator-=(const Tensor& rhs);
    Tensor  operator- (const Tensor& rhs) const;
    // Tensor& operator*=(const Tensor& rhs);
    // Tensor operator* (const Tensor& rhs) const;

    Tensor& operator*=(const int rhs);
    // Tensor operator*(const Tensor& tensor, const int     number);
    // Tensor operator*(const int     number, const Tensor& tensor);
    // functions outside the class

//Other methods
public:
    void set_size(std::uint16_t C, std::uint16_t H, std::uint16_t W);
};

Tensor operator*(const Tensor& tensor, const int     number);
Tensor operator*(const int     number, const Tensor& tensor);


} //namespace etc
