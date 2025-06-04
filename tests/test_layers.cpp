#include "cnn/layers.h"
#include <gtest/gtest.h>

// ReLU层测试
TEST(LayersTest, ReLU) {
  CNN::ReLULayer relu;

  // 创建输入张量
  CNN::Tensor input({2, 3});
  input[0] = -1.0f;
  input[1] = 2.0f;
  input[2] = 0.0f;
  input[3] = 3.0f;
  input[4] = -2.0f;
  input[5] = -0.5f;

  // 运行前向传播
  CNN::Tensor output = relu.forward(input);

  // 检查结果
  EXPECT_FLOAT_EQ(output[0], 0.0f);
  EXPECT_FLOAT_EQ(output[1], 2.0f);
  EXPECT_FLOAT_EQ(output[2], 0.0f);
  EXPECT_FLOAT_EQ(output[3], 3.0f);
  EXPECT_FLOAT_EQ(output[4], 0.0f);
  EXPECT_FLOAT_EQ(output[5], 0.0f);
}

// Sigmoid层测试
TEST(LayersTest, Sigmoid) {
  CNN::SigmoidLayer sigmoid;

  // 创建输入张量
  CNN::Tensor input({2, 2});
  input[0] = 0.0f;
  input[1] = 1.0f;
  input[2] = -1.0f;
  input[3] = 10.0f;

  // 运行前向传播
  CNN::Tensor output = sigmoid.forward(input);

  // 检查结果
  EXPECT_NEAR(output[0], 0.5f, 1e-6);
  EXPECT_NEAR(output[1], 0.7311f, 1e-4);
  EXPECT_NEAR(output[2], 0.2689f, 1e-4);
  EXPECT_NEAR(output[3], 0.9999f, 1e-4);
}

// MaxPool层测试
TEST(LayersTest, MaxPool) {
  CNN::MaxPoolLayer maxpool(2, 2, 0);

  // 创建4x4输入张量
  CNN::Tensor input({1, 1, 4, 4});
  for (size_t i = 0; i < 16; ++i) {
    input[i] = static_cast<float>(i);
  }

  // 运行前向传播
  CNN::Tensor output = maxpool.forward(input);

  // 检查输出形状应该是1x1x2x2
  auto shape = output.shape();
  EXPECT_EQ(shape.size(), 4);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 1);
  EXPECT_EQ(shape[2], 2);
  EXPECT_EQ(shape[3], 2);

  // 检查结果：2x2的最大池化后应该是[[5,7],[13,15]]
  EXPECT_FLOAT_EQ(output[0], 5.0f);
  EXPECT_FLOAT_EQ(output[1], 7.0f);
  EXPECT_FLOAT_EQ(output[2], 13.0f);
  EXPECT_FLOAT_EQ(output[3], 15.0f);
}

// Flatten层测试
TEST(LayersTest, Flatten) {
  CNN::FlattenLayer flatten;

  // 创建形状为[2,3,2]的输入张量
  CNN::Tensor input({2, 3, 2});
  for (size_t i = 0; i < input.size(); ++i) {
    input[i] = static_cast<float>(i);
  }

  // 运行前向传播
  CNN::Tensor output = flatten.forward(input);

  // 检查输出形状应该是[2,6]
  auto shape = output.shape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 6);

  // 检查数据是否正确展平
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_FLOAT_EQ(output[i], static_cast<float>(i));
  }
}

// 主函数
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}