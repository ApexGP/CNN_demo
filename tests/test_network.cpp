#include "cnn/network.h"
#include <gtest/gtest.h>

// 网络创建和基本操作测试
TEST(NetworkTest, Creation) {
  CNN::Network network;

  // 添加一些层
  network.add_relu_layer();
  network.add_sigmoid_layer();

  // 创建输入张量
  CNN::Tensor input({1, 3, 32, 32});
  input.fill(1.0f);

  // 运行前向传播
  CNN::Tensor output = network.forward(input);

  // 由于使用的是ReLU和Sigmoid，输出应该在0-1范围内
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_GE(output[i], 0.0f);
    EXPECT_LE(output[i], 1.0f);
  }
}

// 简单卷积网络测试
TEST(NetworkTest, ConvNet) {
  CNN::Network network;

  // 创建简单的卷积网络
  network.add_conv_layer(16, 3, 1, 1); // 输出: 16x32x32
  network.add_relu_layer();
  network.add_maxpool_layer(2, 2, 0);  // 输出: 16x16x16
  network.add_conv_layer(32, 3, 1, 1); // 输出: 32x16x16
  network.add_relu_layer();
  network.add_maxpool_layer(2, 2, 0); // 输出: 32x8x8
  network.add_flatten_layer();        // 输出: 2048
  network.add_fc_layer(10);           // 输出: 10
  network.add_softmax_layer();

  // 创建输入张量
  CNN::Tensor input({1, 3, 32, 32});
  input.fill(0.5f);

  // 运行前向传播
  CNN::Tensor output = network.forward(input);

  // 检查输出形状
  auto shape = output.shape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 1);
  EXPECT_EQ(shape[1], 10);

  // 检查输出是概率分布
  float sum = 0.0f;
  for (size_t i = 0; i < output.size(); ++i) {
    EXPECT_GE(output[i], 0.0f);
    EXPECT_LE(output[i], 1.0f);
    sum += output[i];
  }
  EXPECT_NEAR(sum, 1.0f, 1e-5);
}

// 训练/评估模式切换测试
TEST(NetworkTest, TrainingMode) {
  CNN::Network network;

  // 添加一个Dropout层（在训练和评估模式下行为不同）
  network.add_dropout_layer(0.5);

  // 创建输入张量
  CNN::Tensor input({1, 10});
  input.fill(1.0f);

  // 在训练模式下，Dropout应该随机将一些元素置为0
  network.train_mode();
  CNN::Tensor train_output = network.forward(input);

  // 在评估模式下，Dropout不应修改输入
  network.eval_mode();
  CNN::Tensor eval_output = network.forward(input);

  // 检查评估模式输出是否与输入相同
  for (size_t i = 0; i < eval_output.size(); ++i) {
    EXPECT_FLOAT_EQ(eval_output[i], input[i]);
  }
}

// 主函数
int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}