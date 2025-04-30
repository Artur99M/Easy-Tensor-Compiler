#include "tensor.hpp"
#include "nn_node.hpp"
#include <gtest/gtest.h>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include "neural_network.hpp"

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

TEST(TestTensor, softmax) {

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

    x.softmax_self();
    std::cout << x.dump();

    Tensor y =
    {
        {   // Batch 0
            {   // Channel 0
                {0.01798620996209156, 0.01798620996209156},
                {0.01798620996209156, 0.01798620996209156}
            },
            {   // Channel 1
                {0.9820137900379085, 0.9820137900379085},
                {0.9820137900379085, 0.9820137900379085}
            }
        },
        {   // Batch 1
            {   // Channel 0
                {0.01798620996209156, 0.01798620996209156},
                {0.01798620996209156, 0.01798620996209156}
            },
            {   // Channel 1
                {0.9820137900379085, 0.9820137900379085},
                {0.9820137900379085, 0.9820137900379085}
            }
        }
    };

    EXPECT_EQ(x, y);
}

TEST(TestNode, evaluateScalarAddOperation) {

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

    Tensor y = {
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

    std::shared_ptr<IWeight> xdata = std::make_shared<IWeight>(x),
                             ydata = std::make_shared<IWeight>(y);


    std::shared_ptr<INode> res = std::make_shared<ScalarAddOperation>(xdata, ydata);

    EXPECT_TRUE(res->is_operation());
    EXPECT_FALSE(res->solved());
    EXPECT_EQ(res->evaluate(), x + y);
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateScalarSubOperation) {

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

    Tensor y = {
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

    std::shared_ptr<IWeight> xdata = std::make_shared<IWeight>(x),
                             ydata = std::make_shared<IWeight>(y);


    std::shared_ptr<INode> res = std::make_shared<ScalarSubOperation>(xdata, ydata);

    EXPECT_TRUE(res->is_operation());
    EXPECT_FALSE(res->solved());
    EXPECT_EQ(res->evaluate(), x - y);
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateMatMulOperation) {

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

    Tensor y = {
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

    std::shared_ptr<IWeight> xdata = std::make_shared<IWeight>(x),
                             ydata = std::make_shared<IWeight>(y);


    std::shared_ptr<INode> res = std::make_shared<MatMulOperation>(xdata, ydata);

    EXPECT_TRUE(res->is_operation());
    EXPECT_FALSE(res->solved());
    EXPECT_EQ(res->evaluate(), x * y);
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateScalarMulOperation) {

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

    std::shared_ptr<IWeight> xdata = std::make_shared<IWeight>(x);


    std::shared_ptr<INode> res = std::make_shared<ScalarMulOperation>(xdata, 2);

    EXPECT_TRUE(res->is_operation());
    EXPECT_FALSE(res->solved());
    EXPECT_EQ(res->evaluate(), x * 2);
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateConvolOperation) {

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

    Tensor y = {
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

    std::shared_ptr<IWeight> xdata = std::make_shared<IWeight>(x),
                             ydata = std::make_shared<IWeight>(y);


    std::shared_ptr<INode> res = std::make_shared<ConvolOperation>(xdata, ydata);

    EXPECT_TRUE(res->is_operation());
    EXPECT_FALSE(res->solved());
    EXPECT_EQ(res->evaluate(), convol(x, y));
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateReLUOperation) {

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

    std::shared_ptr<IWeight> xdata = std::make_shared<IWeight>(x);


    std::shared_ptr<INode> res = std::make_shared<ReLUOperation>(xdata);

    EXPECT_TRUE(res->is_operation());
    EXPECT_FALSE(res->solved());
    EXPECT_EQ(res->evaluate(), x.ReLU());
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateSoftmaxOperation) {

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

    std::shared_ptr<IWeight> xdata = std::make_shared<IWeight>(x);


    std::shared_ptr<INode> res = std::make_shared<SoftmaxOperation>(xdata);

    EXPECT_TRUE(res->is_operation());
    EXPECT_FALSE(res->solved());
    EXPECT_EQ(res->evaluate(), x.softmax());
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateIWeight) {

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

    std::shared_ptr<INode> res = std::make_shared<IWeight>(x);

    EXPECT_FALSE(res->is_operation());
    EXPECT_TRUE(res->solved());
    EXPECT_EQ(res->evaluate(), x);
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateInputData) {

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

    std::shared_ptr<INode> res = std::make_shared<InputData>(x);

    EXPECT_FALSE(res->is_operation());
    EXPECT_TRUE(res->solved());
    EXPECT_EQ(res->evaluate(), x);
    EXPECT_TRUE(res->solved());

}

TEST(TestNode, evaluateINumber) {

    std::shared_ptr<INode> res = std::make_shared<INumber>(2);

    EXPECT_FALSE(res->is_operation());
    EXPECT_TRUE(res->solved());
    EXPECT_THROW(res->evaluate(), std::logic_error);
    EXPECT_TRUE(res->solved());
    EXPECT_EQ(dynamic_cast<INumber&>(*res).val(), 2);

}


TEST(TestTree, MakeTree1) {

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
    std::shared_ptr<INode> node1 = std::make_shared<IWeight>(x),
                           node2 = std::make_shared<InputData>(x),

    node3 = std::make_shared<ScalarAddOperation>(node1, node1),
    node4 = std::make_shared<ScalarMulOperation>(node2, 2);

    std::shared_ptr<INode> root = std::make_shared<ScalarSubOperation>(node3, node4);

    IOperation& rootl = dynamic_cast<IOperation&>(*root);
    std::vector<std::shared_ptr<INode>> rootl_vec = rootl.getArgs();
    EXPECT_EQ(rootl_vec.size(), 2);
    EXPECT_EQ(rootl_vec[0], node3);
    EXPECT_EQ(rootl_vec[1], node4);

    IOperation& node3l = dynamic_cast<IOperation&>(*node3);
    std::vector<std::shared_ptr<INode>> node3l_vec = node3l.getArgs();
    EXPECT_EQ(node3l_vec.size(), 2);
    EXPECT_EQ(node3l_vec[0], node1);
    EXPECT_EQ(node3l_vec[1], node1);

    IOperation& node4l = dynamic_cast<IOperation&>(*node4);
    std::vector<std::shared_ptr<INode>> node4l_vec = node4l.getArgs();
    EXPECT_EQ(node4l_vec.size(), 2);
    EXPECT_EQ(node4l_vec[0], node2);
    EXPECT_EQ(typeid(*(node4l_vec[1])), typeid(INumber));
    EXPECT_EQ(dynamic_cast<INumber&>(*(node4l_vec[1])).val(), 2);

}

TEST(TestTree, MakeTree2) {

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
    std::shared_ptr<INode> node1 = std::make_shared<IWeight>(x),
                           node2 = std::make_shared<InputData>(x),
                           node3 = std::make_shared<IWeight>(x),

    node4 = std::make_shared<MatMulOperation>(node1, node2),
    node5 = std::make_shared<ConvolOperation>(node4, node3);

    std::shared_ptr<INode> root = std::make_shared<SoftmaxOperation>(node5);

    IOperation& rootl = dynamic_cast<IOperation&>(*root);
    std::vector<std::shared_ptr<INode>> rootl_vec = rootl.getArgs();
    EXPECT_EQ(rootl_vec.size(), 1);
    EXPECT_EQ(rootl_vec[0], node5);
    EXPECT_EQ(typeid(rootl), typeid(SoftmaxOperation));

    IOperation& node5l = dynamic_cast<IOperation&>(*node5);
    std::vector<std::shared_ptr<INode>> node5l_vec = node5l.getArgs();
    EXPECT_EQ(node5l_vec.size(), 2);
    EXPECT_EQ(node5l_vec[0], node4);
    EXPECT_EQ(node5l_vec[1], node3);
    EXPECT_EQ(typeid(node5l), typeid(ConvolOperation));

    EXPECT_EQ(typeid(*node3), typeid(IWeight));

    IOperation& node4l = dynamic_cast<IOperation&>(*node4);
    std::vector<std::shared_ptr<INode>> node4l_vec = node4l.getArgs();
    EXPECT_EQ(node4l_vec.size(), 2);
    EXPECT_EQ(node4l_vec[0], node1);
    EXPECT_EQ(node4l_vec[1], node2);
    EXPECT_EQ(typeid(node4l), typeid(MatMulOperation));

    EXPECT_EQ(typeid(*node1), typeid(IWeight));

    EXPECT_EQ(typeid(*node2), typeid(InputData));

}

TEST(TestTree, MakeTree3) {

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
    std::shared_ptr<INode> node1 = std::make_shared<IWeight>(x),
                           node2 = std::make_shared<InputData>(x),
                           node3 = std::make_shared<IWeight>(x),

    node4 = std::make_shared<MatMulOperation>(node1, node2),
    node5 = std::make_shared<ConvolOperation>(node4, node3);

    std::shared_ptr<INode> root = std::make_shared<SoftmaxOperation>(node5);

    IOperation& rootl = dynamic_cast<IOperation&>(*root);
    std::vector<std::shared_ptr<INode>> rootl_vec = rootl.getArgs();
    EXPECT_EQ(rootl_vec.size(), 1);
    EXPECT_EQ(rootl_vec[0], node5);
    EXPECT_EQ(typeid(rootl), typeid(SoftmaxOperation));

    IOperation& node5l = dynamic_cast<IOperation&>(*node5);
    std::vector<std::shared_ptr<INode>> node5l_vec = node5l.getArgs();
    EXPECT_EQ(node5l_vec.size(), 2);
    EXPECT_EQ(node5l_vec[0], node4);
    EXPECT_EQ(node5l_vec[1], node3);
    EXPECT_EQ(typeid(node5l), typeid(ConvolOperation));

    EXPECT_EQ(typeid(*node3), typeid(IWeight));

    IOperation& node4l = dynamic_cast<IOperation&>(*node4);
    std::vector<std::shared_ptr<INode>> node4l_vec = node4l.getArgs();
    EXPECT_EQ(node4l_vec.size(), 2);
    EXPECT_EQ(node4l_vec[0], node1);
    EXPECT_EQ(node4l_vec[1], node2);
    EXPECT_EQ(typeid(node4l), typeid(MatMulOperation));

    EXPECT_EQ(typeid(*node1), typeid(IWeight));

    EXPECT_EQ(typeid(*node2), typeid(InputData));

}

TEST(TestTree, MakeTree4) {

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
    std::shared_ptr<INode> node1 = std::make_shared<IWeight>(x),
                           node2 = std::make_shared<InputData>(x),
                           node3 = std::make_shared<IWeight>(x),
                           node4 = std::make_shared<InputData>(x),

    node5 = std::make_shared<MatMulOperation>(node1, node2),
    node6 = std::make_shared<ConvolOperation>(node4, node3),
    node7 = std::make_shared<ReLUOperation>(node5),
    node8 = std::make_shared<ScalarMulOperation>(node6, 7);

    EXPECT_EQ(typeid(*node1), typeid(IWeight));
    EXPECT_EQ(typeid(*node2), typeid(InputData));
    EXPECT_EQ(typeid(*node3), typeid(IWeight));
    EXPECT_EQ(typeid(*node4), typeid(InputData));

    std::shared_ptr<INode> root = std::make_shared<ScalarSubOperation>(node8, node7);

    IOperation& rootl = dynamic_cast<IOperation&>(*root);
    std::vector<std::shared_ptr<INode>> rootl_vec = rootl.getArgs();
    EXPECT_EQ(rootl_vec.size(), 2);
    EXPECT_EQ(rootl_vec[0], node8);
    EXPECT_EQ(rootl_vec[1], node7);
    EXPECT_EQ(typeid(rootl), typeid(ScalarSubOperation));

    IOperation& node8l = dynamic_cast<IOperation&>(*node8);
    std::vector<std::shared_ptr<INode>> node8l_vec = node8l.getArgs();
    EXPECT_EQ(node8l_vec.size(), 2);
    EXPECT_EQ(node8l_vec[0], node6);
    EXPECT_EQ(typeid(*(node8l_vec[1])), typeid(INumber));
    EXPECT_EQ(dynamic_cast<INumber&>(*(node8l_vec[1])).val(), 7);
    EXPECT_EQ(typeid(node8l), typeid(ScalarMulOperation));

    IOperation& node7l = dynamic_cast<IOperation&>(*node7);
    std::vector<std::shared_ptr<INode>> node7l_vec = node7l.getArgs();
    EXPECT_EQ(node7l_vec.size(), 1);
    EXPECT_EQ(node7l_vec[0], node5);
    EXPECT_EQ(typeid(node7l), typeid(ReLUOperation));

    IOperation& node5l = dynamic_cast<IOperation&>(*node5);
    std::vector<std::shared_ptr<INode>> node5l_vec = node5l.getArgs();
    EXPECT_EQ(node5l_vec.size(), 2);
    EXPECT_EQ(node5l_vec[0], node1);
    EXPECT_EQ(node5l_vec[1], node2);
    EXPECT_EQ(typeid(node5l), typeid(MatMulOperation));

    IOperation& node6l = dynamic_cast<IOperation&>(*node6);
    std::vector<std::shared_ptr<INode>> node6l_vec = node6l.getArgs();
    EXPECT_EQ(node6l_vec.size(), 2);
    EXPECT_EQ(node6l_vec[0], node4);
    EXPECT_EQ(node6l_vec[1], node3);
    EXPECT_EQ(typeid(node6l), typeid(ConvolOperation));
}

TEST(TestNN, RightTree) {
    NeuralNetwork nn;

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
    std::shared_ptr<INode> node1 = std::make_shared<IWeight>(x),
                           node2 = std::make_shared<InputData>(x),
                           node3 = std::make_shared<IWeight>(x),
                           node4 = std::make_shared<InputData>(x),

    node5 = nn.addOp(std::make_shared<MatMulOperation>(node1, x)),
    node6 = nn.addOp(std::make_shared<ConvolOperation>(x, node3)),
    node7 = nn.addOp(std::make_shared<ReLUOperation>(node5)),
    node8 = nn.addOp(std::make_shared<ScalarMulOperation>(node6, 7));

    EXPECT_EQ(typeid(*node1), typeid(IWeight));
    EXPECT_EQ(typeid(*node2), typeid(InputData));
    EXPECT_EQ(typeid(*node3), typeid(IWeight));
    EXPECT_EQ(typeid(*node4), typeid(InputData));

    std::shared_ptr<INode> root = nn.addOp(std::make_shared<ScalarSubOperation>(node8, node7));
    root = nn.infer_node();


    IOperation& rootl = dynamic_cast<IOperation&>(*root);
    std::vector<std::shared_ptr<INode>> rootl_vec = rootl.getArgs();
    EXPECT_EQ(rootl_vec.size(), 2);
    EXPECT_EQ(rootl_vec[0], node8);
    EXPECT_EQ(rootl_vec[1], node7);
    EXPECT_EQ(typeid(rootl), typeid(ScalarSubOperation));

    IOperation& node8l = dynamic_cast<IOperation&>(*node8);
    std::vector<std::shared_ptr<INode>> node8l_vec = node8l.getArgs();
    EXPECT_EQ(node8l_vec.size(), 2);
    EXPECT_EQ(node8l_vec[0], node6);
    EXPECT_EQ(typeid(*(node8l_vec[1])), typeid(INumber));
    EXPECT_EQ(dynamic_cast<INumber&>(*(node8l_vec[1])).val(), 7);
    EXPECT_EQ(typeid(node8l), typeid(ScalarMulOperation));

    IOperation& node7l = dynamic_cast<IOperation&>(*node7);
    std::vector<std::shared_ptr<INode>> node7l_vec = node7l.getArgs();
    EXPECT_EQ(node7l_vec.size(), 1);
    EXPECT_EQ(node7l_vec[0], node5);
    EXPECT_EQ(typeid(node7l), typeid(ReLUOperation));

    IOperation& node5l = dynamic_cast<IOperation&>(*node5);
    std::vector<std::shared_ptr<INode>> node5l_vec = node5l.getArgs();
    EXPECT_EQ(node5l_vec.size(), 2);
    EXPECT_EQ(node5l_vec[0], node1);
    EXPECT_EQ(typeid(*(node5l_vec[1])), typeid(IWeight));
    EXPECT_EQ(node5l_vec[1]->evaluate(), x);
    EXPECT_EQ(typeid(node5l), typeid(MatMulOperation));

    IOperation& node6l = dynamic_cast<IOperation&>(*node6);
    std::vector<std::shared_ptr<INode>> node6l_vec = node6l.getArgs();
    EXPECT_EQ(node6l_vec.size(), 2);
    EXPECT_EQ(typeid(*(node6l_vec[0])), typeid(IWeight));
    EXPECT_EQ(node6l_vec[1]->evaluate(), x);
    EXPECT_EQ(node6l_vec[1], node3);
    EXPECT_EQ(typeid(node6l), typeid(ConvolOperation));
}
//
// TEST(TestNN, RightTreeWithInfer) {
//     NeuralNetwork nn;
//
//     Tensor x = {
//                {
//                {
//                {1, 2},
//                {3, 4}
//                },
//                {
//                {5, 6},
//                {7, 8}
//                }
//                },
//                {
//                {
//                {9, 10},
//                {11, 12}
//                },
//                {
//                {13, 14},
//                {15, 16}
//                }
//                }
//                };
//     std::shared_ptr<INode> node1 = std::make_shared<IWeight>(x),
//                            node2 = std::make_shared<InputData>(x),
//                            node3 = std::make_shared<IWeight>(x),
//                            node4 = std::make_shared<InputData>(x),
//
//     node5 = nn.addOp(std::make_shared<MatMulOperation>(node1, x)),
//     node6 = nn.addOp(std::make_shared<ConvolOperation>(x, node3)),
//     node7 = nn.addOp(std::make_shared<ReLUOperation>(node5)),
//     node8 = nn.addOp(std::make_shared<ScalarMulOperation>(node6, 7)),
//     node9 = nn.addOp(std::make_shared<MatMulOperation>(node5, nn.infer_node()));
//
//     EXPECT_EQ(typeid(*node1), typeid(IWeight));
//     EXPECT_EQ(typeid(*node2), typeid(InputData));
//     EXPECT_EQ(typeid(*node3), typeid(IWeight));
//     EXPECT_EQ(typeid(*node4), typeid(InputData));
//
//     std::shared_ptr<INode> root = nn.addOp(std::make_shared<ScalarSubOperation>(node8, node7));
//     root = nn.infer_node();
//
//
//     IOperation& rootl = dynamic_cast<IOperation&>(*root);
//     std::vector<std::shared_ptr<INode>> rootl_vec = rootl.getArgs();
//     EXPECT_EQ(rootl_vec.size(), 2);
//     EXPECT_EQ(rootl_vec[0], node8);
//     EXPECT_EQ(rootl_vec[1], node7);
//     EXPECT_EQ(typeid(rootl), typeid(ScalarSubOperation));
//
//     IOperation& node8l = dynamic_cast<IOperation&>(*node8);
//     std::vector<std::shared_ptr<INode>> node8l_vec = node8l.getArgs();
//     EXPECT_EQ(node8l_vec.size(), 2);
//     EXPECT_EQ(node8l_vec[0], node6);
//     EXPECT_EQ(typeid(*(node8l_vec[1])), typeid(INumber));
//     EXPECT_EQ(dynamic_cast<INumber&>(*(node8l_vec[1])).val(), 7);
//     EXPECT_EQ(typeid(node8l), typeid(ScalarMulOperation));
//
//     IOperation& node7l = dynamic_cast<IOperation&>(*node7);
//     std::vector<std::shared_ptr<INode>> node7l_vec = node7l.getArgs();
//     EXPECT_EQ(node7l_vec.size(), 1);
//     EXPECT_EQ(node7l_vec[0], node5);
//     EXPECT_EQ(typeid(node7l), typeid(ReLUOperation));
//
//     IOperation& node5l = dynamic_cast<IOperation&>(*node5);
//     std::vector<std::shared_ptr<INode>> node5l_vec = node5l.getArgs();
//     EXPECT_EQ(node5l_vec.size(), 2);
//     EXPECT_EQ(node5l_vec[0], node1);
//     EXPECT_EQ(node5l_vec[1], node2);
//     EXPECT_EQ(typeid(node5l), typeid(MatMulOperation));
//
//     IOperation& node6l = dynamic_cast<IOperation&>(*node6);
//     std::vector<std::shared_ptr<INode>> node6l_vec = node6l.getArgs();
//     EXPECT_EQ(node6l_vec.size(), 2);
//     EXPECT_EQ(node6l_vec[0], node4);
//     EXPECT_EQ(node6l_vec[1], node3);
//     EXPECT_EQ(typeid(node6l), typeid(ConvolOperation));
//
//     IOperation& node9l = dynamic_cast<IOperation&>(*node9);
//     std::vector<std::shared_ptr<INode>> node9l_vec = node9l.getArgs();
//     EXPECT_EQ(node9l_vec.size(), 2);
//     EXPECT_EQ(node9l_vec[0], node5);
//     EXPECT_EQ(node9l_vec[1], node8);
//     EXPECT_EQ(typeid(node9l), typeid(MatMulOperation));
//
//     EXPECT_EQ(root, node9);
// }
//
