#pragma once
#include <vector>
#include <stdexcept>
#include <cstdint>
#include <string>

namespace etc {

class Tensor {

private:
    // unsigned N_ = 1;
    unsigned C_ = 0,
             H_ = 0,
             W_ = 0;
    std::vector<std::vector<std::vector<int>>> channels_;

public:
    Tensor() = default;
    Tensor(unsigned C, unsigned H, unsigned W);

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
    inline      std::vector<std::vector<int>>&  operator[](unsigned i) {
        return channels_[i];
    }
    inline const std::vector<std::vector<int>>& operator[](unsigned i) const {
        return channels_[i];
    }
    inline      int&                            at        (unsigned i, unsigned j, unsigned k) {
        if (i >= C_ || j >= H_ || k >= W_)
            throw std::out_of_range("in etc::Tensor::at");
        return channels_[i][j][k];
    }
    inline const int&                           at        (unsigned i, unsigned j, unsigned k) const {
        if (i >= C_ || j >= H_ || k >= W_)
            throw std::out_of_range("in etc::Tensor::at");
        return channels_[i][j][k];
    }

//size
public:
    inline unsigned C() const {
        return C_;
    }
    inline unsigned H() const {
        return H_;
    }
    inline unsigned W() const {
        return W_;
    }

//Math operations
public:
    Tensor& operator+=(const Tensor& rhs);
    Tensor  operator+ (const Tensor& rhs) const;
    Tensor& operator-=(const Tensor& rhs);
    Tensor  operator- (const Tensor& rhs) const;
    Tensor& operator*=(const Tensor& rhs);
    Tensor  operator* (const Tensor& rhs) const;
    Tensor  transpose () const;
    Tensor& operator*=(const int rhs);

    // functions outside the class
    // Tensor operator*(const Tensor& tensor, const int     number);
    // Tensor operator*(const int     number, const Tensor& tensor);
    // Tensor scal_mul (const Tensor& lhs   , const Tensor& rhs);

//Other methods
public:
    void        set_size(unsigned C, unsigned H, unsigned W);
    std::string dump    () const;
};

Tensor operator*(const Tensor& tensor, const int     number);
Tensor operator*(const int     number, const Tensor& tensor);
Tensor scal_mul (const Tensor& lhs   , const Tensor& rhs);


} //namespace etc
