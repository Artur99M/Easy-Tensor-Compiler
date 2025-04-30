#pragma once
#include <vector>
#include <stdexcept>
#include <string>

namespace etc {

class Tensor {

private:

//classes for simpler handing
    struct line {
        double* data;
        inline double& operator[](const size_t l) {
            return data[l];
        }
        inline const double& operator[](const size_t l) const {
            return data[l];
        }
    };
    struct matrix {
        double* data;
        unsigned W;
        inline line operator[](const size_t l) {
            return {data + l * W};
        }
        inline const line operator[](const size_t l) const {
            return {data + l * W};
        }
    };
    struct channel {
        double* data;
        inline matrix operator[](const size_t c) {
            return {data + c * W * H, W};
        }
        inline const matrix operator[](const size_t c) const {
            return {data + c * W * H, W};
        }
        unsigned H, W;
    };

// members
    unsigned N_ = 1; //batches
    unsigned C_ = 0,
             H_ = 0,
             W_ = 0;

    // here vector used for not thinking about memory
    std::vector<double> data_ = std::vector<double>(N_ * C_ * H_ * W_);

public:
//ctors, dtors, operator=
    Tensor() = default;
    Tensor(const unsigned N, const unsigned C, const unsigned H, const unsigned W);

    Tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<double>>>> init);
    Tensor(const Tensor& other) = default;
    Tensor(Tensor&& other);
    Tensor& operator=(const Tensor& rhs) = default;
    Tensor& operator=(Tensor&& rhs);
    ~Tensor() = default;

//handing element
public:
    inline       channel operator[](const size_t i);
    inline const channel operator[](const size_t i)                     const;
    inline       double&    at     (const unsigned b, const unsigned i,
                                    const unsigned j, const unsigned k);
    inline const double&    at     (const unsigned b, const unsigned i,
                                    const unsigned j, const unsigned k) const;

//size
public:
    inline unsigned N() const {return N_;}
    inline unsigned C() const {return C_;}
    inline unsigned H() const {return H_;}
    inline unsigned W() const {return W_;}

    void set_size(const unsigned N, const unsigned C, const unsigned H, const unsigned W);

//Math operations
public:
    bool    operator==  (const Tensor& rhs) const;
    bool    operator!=  (const Tensor& rhs) const;
    Tensor& operator+=  (const Tensor& rhs)      ;
    Tensor  operator+   (const Tensor& rhs) const;
    Tensor& operator-=  (const Tensor& rhs)      ;
    Tensor  operator-   (const Tensor& rhs) const;
    Tensor& operator*=  (const Tensor& rhs)      ;
    Tensor  operator*   (const Tensor& rhs) const;
    Tensor& operator*=  (const double  num)      ;

    Tensor  transpose   ()                  const;
    Tensor  ReLU        ()                  const;
    Tensor& ReLU_self   ()                       ;
    Tensor  softmax     ()                  const;
    Tensor& softmax_self()                       ;

    // math functions outside the class
    // Tensor operator*(const Tensor& tensor, const double  number);
    // Tensor operator*(const double  number, const Tensor& tensor);
    // Tensor convol   (const Tensor& lhs   , const Tensor& rhs);

//Other methods
public:
                 std::string          dump     () const;
                 std::string          dump_init() const;
    inline       std::vector<double>& data     ()       {return data_;}
    inline const std::vector<double>& data     () const {return data_;}
};

Tensor operator*(const Tensor& tensor, const double  number);
Tensor operator*(const double  number, const Tensor& tensor);
Tensor convol   (const Tensor& lhs   , const Tensor& rhs);



// definitions for inline funcs

inline       Tensor::channel Tensor::operator[](const size_t i) {
    return {data_.data() + i * W_ * H_ * C_, H_, W_};
}
inline const Tensor::channel Tensor::operator[](const size_t i) const {
    return {const_cast<double*>(data_.data() + i * W_ * H_ * C_), H_, W_};
}
inline       double&            Tensor::at        (const unsigned b, const unsigned i, const unsigned j, const unsigned k) {
    if (b >= N_ || i >= C_ || j >= H_ || k >= W_)
        throw std::out_of_range("in etc::Tensor::at");

    channel c = {(data_.data() + b * W_ * H_ * C_), H_, W_};
    return c[i][j][k];
}
inline const double&            Tensor::at        (const unsigned b, const unsigned i, const unsigned j, const unsigned k) const {
    if (b >= N_ || i >= C_ || j >= H_ || k >= W_)
        throw std::out_of_range("in etc::Tensor::at");

    channel c = {const_cast<double*>(data_.data() + b * W_ * H_ * C_), H_, W_};
    return c[i][j][k];
}
} //namespace etc
