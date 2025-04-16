#include "tensor.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>

using namespace etc;

TEST(TesorTest, CtorInitializerList) {
    Tensor tensor = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    auto data = tensor.data();
    EXPECT_EQ(data[0 ], 1  );
    EXPECT_EQ(data[1 ], 2  );
    EXPECT_EQ(data[2 ], 3  );
    EXPECT_EQ(data[3 ], 4  );
    EXPECT_EQ(data[4 ], 5  );
    EXPECT_EQ(data[5 ], 6  );
    EXPECT_EQ(data[6 ], 7  );
    EXPECT_EQ(data[7 ], 8  );
    EXPECT_EQ(data[8 ], 9  );
    EXPECT_EQ(data[9 ], 10 );
    EXPECT_EQ(data[10], 11 );
    EXPECT_EQ(data[11], 12 );
    EXPECT_EQ(data[12], 13 );
    EXPECT_EQ(data[13], 14 );
    EXPECT_EQ(data[14], 15 );
    EXPECT_EQ(data[15], 16 );

    EXPECT_EQ(tensor.N(), 2);
    EXPECT_EQ(tensor.C(), 2);
    EXPECT_EQ(tensor.H(), 2);
    EXPECT_EQ(tensor.W(), 2);

}

TEST(TesorTest, CtorMove) {
    Tensor tensor = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    Tensor tensor2 = std::move(tensor);
    auto data = tensor2.data();


    EXPECT_EQ(data[0 ], 1  );
    EXPECT_EQ(data[1 ], 2  );
    EXPECT_EQ(data[2 ], 3  );
    EXPECT_EQ(data[3 ], 4  );
    EXPECT_EQ(data[4 ], 5  );
    EXPECT_EQ(data[5 ], 6  );
    EXPECT_EQ(data[6 ], 7  );
    EXPECT_EQ(data[7 ], 8  );
    EXPECT_EQ(data[8 ], 9  );
    EXPECT_EQ(data[9 ], 10 );
    EXPECT_EQ(data[10], 11 );
    EXPECT_EQ(data[11], 12 );
    EXPECT_EQ(data[12], 13 );
    EXPECT_EQ(data[13], 14 );
    EXPECT_EQ(data[14], 15 );
    EXPECT_EQ(data[15], 16 );

    EXPECT_EQ(tensor2.N(), 2);
    EXPECT_EQ(tensor2.C(), 2);
    EXPECT_EQ(tensor2.H(), 2);
    EXPECT_EQ(tensor2.W(), 2);

    EXPECT_EQ(tensor.data().size(), 0);
    EXPECT_EQ(tensor.N(), 0);
    EXPECT_EQ(tensor.C(), 0);
    EXPECT_EQ(tensor.H(), 0);
    EXPECT_EQ(tensor.W(), 0);
}

TEST(TesorTest, OperatorAdd) {

    Tensor x = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    Tensor z = {
               {
               {
               {-4, -3},
               {2, 1}
               },
               {
               {-8, -7},
               {6, 5}
               }
               },
               {
               {
               {-12, -11},
               {10, 9}
               },
               {
               {-16, -15},
               {14, 13}
               }
               }
               };

    Tensor res = x + z;

    auto data = res.data();
    EXPECT_EQ(data[0 ], -3 );
    EXPECT_EQ(data[1 ], -1 );
    EXPECT_EQ(data[2 ], 5  );
    EXPECT_EQ(data[3 ], 5  );
    EXPECT_EQ(data[4 ], -3 );
    EXPECT_EQ(data[5 ], -1 );
    EXPECT_EQ(data[6 ], 13 );
    EXPECT_EQ(data[7 ], 13 );
    EXPECT_EQ(data[8 ], -3 );
    EXPECT_EQ(data[9 ], -1 );
    EXPECT_EQ(data[10], 21 );
    EXPECT_EQ(data[11], 21 );
    EXPECT_EQ(data[12], -3 );
    EXPECT_EQ(data[13], -1 );
    EXPECT_EQ(data[14], 29 );
    EXPECT_EQ(data[15], 29 );

    EXPECT_EQ(res.N(), 2);
    EXPECT_EQ(res.C(), 2);
    EXPECT_EQ(res.H(), 2);
    EXPECT_EQ(res.W(), 2);
}

TEST(TesorTest, OperatorSub) {

    Tensor x = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    Tensor res = x - x;

    auto data = res.data();
    EXPECT_EQ(data[0 ], 0 );
    EXPECT_EQ(data[1 ], 0 );
    EXPECT_EQ(data[2 ], 0 );
    EXPECT_EQ(data[3 ], 0 );
    EXPECT_EQ(data[4 ], 0 );
    EXPECT_EQ(data[5 ], 0 );
    EXPECT_EQ(data[6 ], 0 );
    EXPECT_EQ(data[7 ], 0 );
    EXPECT_EQ(data[8 ], 0 );
    EXPECT_EQ(data[9 ], 0 );
    EXPECT_EQ(data[10], 0 );
    EXPECT_EQ(data[11], 0 );
    EXPECT_EQ(data[12], 0 );
    EXPECT_EQ(data[13], 0 );
    EXPECT_EQ(data[14], 0 );
    EXPECT_EQ(data[15], 0 );

    EXPECT_EQ(res.N(), 2);
    EXPECT_EQ(res.C(), 2);
    EXPECT_EQ(res.H(), 2);
    EXPECT_EQ(res.W(), 2);
}


TEST(TesorTest, OperatorMul1) {

    Tensor x = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    Tensor E = {
               {
               {
               {0, 1},
               {1, 0}
               },
               {
               {0, 1},
               {1, 0}
               }
               },
               {
               {
               {0, 1},
               {1, 0}
               },
               {
               {0, 1},
               {1, 0}
               }
               }
               };
    Tensor res = E * x * E;

    auto data = res.data();
    EXPECT_EQ(data[0 ], 4 );
    EXPECT_EQ(data[1 ], 3 );
    EXPECT_EQ(data[2 ], 2 );
    EXPECT_EQ(data[3 ], 1 );
    EXPECT_EQ(data[4 ], 8 );
    EXPECT_EQ(data[5 ], 7 );
    EXPECT_EQ(data[6 ], 6 );
    EXPECT_EQ(data[7 ], 5 );
    EXPECT_EQ(data[8 ], 12);
    EXPECT_EQ(data[9 ], 11);
    EXPECT_EQ(data[10], 10);
    EXPECT_EQ(data[11], 9 );
    EXPECT_EQ(data[12], 16);
    EXPECT_EQ(data[13], 15);
    EXPECT_EQ(data[14], 14);
    EXPECT_EQ(data[15], 13);

    EXPECT_EQ(res.N(), 2);
    EXPECT_EQ(res.C(), 2);
    EXPECT_EQ(res.H(), 2);
    EXPECT_EQ(res.W(), 2);
}

TEST(TesorTest, OperatorMul2) {

    Tensor x = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };
    Tensor str = {
               {
               {
               {1, 1}
               },
               {
               {1, 1},
               }
               },
               {
               {
               {1, 1}
               },
               {
               {1, 1}
               }
               }
               };

    auto res  = str * x;
    auto data = res.data();

    EXPECT_EQ(data[0 ], 4 );
    EXPECT_EQ(data[1 ], 6 );
    EXPECT_EQ(data[2 ], 12);
    EXPECT_EQ(data[3 ], 14);
    EXPECT_EQ(data[4 ], 20);
    EXPECT_EQ(data[5 ], 22);
    EXPECT_EQ(data[6 ], 28);
    EXPECT_EQ(data[7 ], 30);

    EXPECT_EQ(data.size(), 8);
    EXPECT_EQ(res.N(), 2);
    EXPECT_EQ(res.C(), 2);
    EXPECT_EQ(res.H(), 1);
    EXPECT_EQ(res.W(), 2);
}

TEST(TesorTest, OperatorMulNumber) {

    Tensor x = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    Tensor res = x * 5;

    EXPECT_EQ(res.N(), x.N());
    EXPECT_EQ(res.C(), x.C());
    EXPECT_EQ(res.H(), x.H());
    EXPECT_EQ(res.W(), x.W());

    for (unsigned b = 0; b < x.N(); ++b)
        for (unsigned c = 0; c < x.C(); ++c)
            for (unsigned h = 0; h < x.H(); ++h)
                for (unsigned w = 0; w < x.W(); ++w)
                    EXPECT_EQ(res[b][c][h][w], x[b][c][h][w] * 5);

}

TEST(TesorTest, OperatorSquareBrackets) {

    Tensor x = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    auto& data = x.data();
    size_t sz = data.size();
    unsigned N = x.N(), C = x.C(), H = x.H(), W = x.W();
    for (size_t i = 0; i < sz; ++i) {
        EXPECT_TRUE(i < N * C * H * W);
        unsigned b = i / (C * H * W),
                 c = (i % (C * H * W)) / (H * W),
                 h = (i % (H * W)) / W,
                 w = i % W;
        EXPECT_EQ(data[i], x[b][c][h][w]);
    }


}

TEST(TesorTest, ReLU) {
    Tensor z = {
               {
               {
               {-4, -3},
               {2, 1}
               },
               {
               {-8, -7},
               {6, 5}
               }
               },
               {
               {
               {-12, -11},
               {10, 9}
               },
               {
               {-16, -15},
               {14, 13}
               }
               }
               };

    Tensor res = z.ReLU();

    for (size_t i = 0; i < 16; ++i)
        EXPECT_TRUE(z.data()[i] > 0 ? z.data()[i] == res.data()[i] : res.data()[i] == 0);
}

TEST(TesorTest, Transpose) {
    Tensor z = {
               {
               {
               {-4, -3}
               },
               {
               {-8, -7}
               }
               },
               {
               {
               {-12, -11}
               },
               {
               {-16, -15}
               }
               }
               };

    Tensor res = z.transpose();

    EXPECT_EQ(res.N(), z.N());
    EXPECT_EQ(res.C(), z.C());
    EXPECT_EQ(res.W(), z.H());
    EXPECT_EQ(res.H(), z.W());

    for (unsigned b = 0; b < z.N(); ++b)
        for (unsigned c = 0; c < z.C(); ++c)
            for (unsigned h = 0; h < z.H(); ++h)
                for (unsigned w = 0; w < z.W(); ++w)
                    EXPECT_EQ(res[b][c][w][h], z[b][c][h][w]);


    res = res.transpose();

    EXPECT_EQ(res.N(), z.N());
    EXPECT_EQ(res.C(), z.C());
    EXPECT_EQ(res.H(), z.H());
    EXPECT_EQ(res.W(), z.W());

    for (unsigned b = 0; b < z.N(); ++b)
        for (unsigned c = 0; c < z.C(); ++c)
            for (unsigned h = 0; h < z.H(); ++h)
                for (unsigned w = 0; w < z.W(); ++w)
                    EXPECT_EQ(res[b][c][h][w], z[b][c][h][w]);

}

TEST(TesorTest, Convol) {
    Tensor x = {
               {
               {
               {1, 2},
               {3, 4}
               },
               {
               {5, 6},
               {7, 8}
               }
               },
               {
               {
               {9, 10},
               {11, 12}
               },
               {
               {13, 14},
               {15, 16}
               }
               }
               };

    Tensor z = {
               {
               {
               {-4, -3},
               {2, 1}
               },
               {
               {-8, -7},
               {6, 5}
               }
               },
               {
               {
               {-12, -11},
               {10, 9}
               },
               {
               {-16, -15},
               {14, 13}
               }
               }
               };

    Tensor res = convol(x, z);

    EXPECT_EQ(res.N(), 2);
    EXPECT_EQ(res.C(), 2);
    EXPECT_EQ(res.H(), 1);
    EXPECT_EQ(res.W(), 1);
    EXPECT_EQ(res.data().size(), 4);

    for (int x : res.data())
        EXPECT_EQ(x, 0);


}
