#include "cnn/tensor.h"
#include <cassert>
#include <cmath> // 添加数学函数头文件
#include <gtest/gtest.h>
#include <iostream>
#include <vector>

using namespace CNN;

// 简单的测试助手函数
#define TEST_ASSERT(condition, message)                                        \
  do {                                                                         \
    if (!(condition)) {                                                        \
      std::cerr << "测试失败: " << message << std::endl;                       \
      std::cerr << "位置: " << __FILE__ << ":" << __LINE__ << std::endl;       \
      return false;                                                            \
    }                                                                          \
  } while (0)

// 测试Tensor创建和基本属性
bool test_tensor_creation() {
  std::cout << "测试Tensor创建和基本属性..." << std::endl;

  // 创建张量
  std::vector<size_t> shape = {2, 3}; // 改为size_t类型
  Tensor tensor(shape);

  // 检查形状和大小
  TEST_ASSERT(tensor.ndim() == 2, "维度应该为2");
  TEST_ASSERT(tensor.size() == 6, "大小应该为6");
  TEST_ASSERT(tensor.shape()[0] == 2, "第一维应为2");
  TEST_ASSERT(tensor.shape()[1] == 3, "第二维应为3");

  // 检查数据指针
  TEST_ASSERT(tensor.data() != nullptr, "数据指针不应为空");

  // 检查初始值
  for (size_t i = 0; i < tensor.size(); i++) {
    TEST_ASSERT(tensor[i] == 0.0f, "初始值应为0"); // 使用[]索引代替at()
  }

  return true;
}

// 测试数据访问和修改
bool test_tensor_data_access() {
  std::cout << "测试数据访问和修改..." << std::endl;

  Tensor tensor({2, 3});

  // 测试[]方法
  tensor[0] = 1.0f;
  tensor[1] = 2.0f;
  TEST_ASSERT(tensor[0] == 1.0f, "值应为1.0");
  TEST_ASSERT(tensor[1] == 2.0f, "值应为2.0");

  // 测试多维索引
  tensor.set({0, 1}, 3.0f);
  tensor.set({1, 2}, 4.0f);
  TEST_ASSERT(tensor.get({0, 1}) == 3.0f, "值应为3.0");
  TEST_ASSERT(tensor.get({1, 2}) == 4.0f, "值应为4.0");

  return true;
}

// 测试初始化方法
bool test_tensor_initialization() {
  std::cout << "测试初始化方法..." << std::endl;

  // 测试zeros
  Tensor tensor1({2, 3});
  tensor1.zeros();
  for (size_t i = 0; i < tensor1.size(); i++) {
    TEST_ASSERT(tensor1[i] == 0.0f, "zeros应将所有值设为0");
  }

  // 测试ones
  Tensor tensor2({2, 3});
  tensor2.ones();
  for (size_t i = 0; i < tensor2.size(); i++) {
    TEST_ASSERT(tensor2[i] == 1.0f, "ones应将所有值设为1");
  }

  // 测试rand初始化（只能检查范围）
  Tensor tensor3({10, 10});
  float low = 0.0f, high = 1.0f;
  tensor3.rand(low, high); // 使用rand方法代替uniform
  for (size_t i = 0; i < tensor3.size(); i++) {
    TEST_ASSERT(tensor3[i] >= low && tensor3[i] <= high,
                "rand应在范围内生成值");
  }

  return true;
}

// 测试数学运算
bool test_tensor_math_ops() {
  std::cout << "测试数学运算..." << std::endl;

  // 创建测试张量
  Tensor a({2, 3});
  Tensor b({2, 3});

  // 设置值
  for (size_t i = 0; i < a.size(); i++) {
    a[i] = 1.0f;
    b[i] = 2.0f;
  }

  // 测试加法
  Tensor c = a + b;
  TEST_ASSERT(c.size() == a.size(), "加法结果大小应与输入相同");
  for (size_t i = 0; i < c.size(); i++) {
    TEST_ASSERT(c[i] == 3.0f, "1.0 + 2.0 应等于 3.0");
  }

  // 测试标量乘法
  Tensor d = a * 3.0f;
  for (size_t i = 0; i < d.size(); i++) {
    TEST_ASSERT(d[i] == 3.0f, "1.0 * 3.0 应等于 3.0");
  }

  return true;
}

// 测试矩阵乘法
bool test_tensor_matmul() {
  std::cout << "测试矩阵乘法..." << std::endl;

  // 创建矩阵 A(2x3) 和 B(3x2)
  Tensor a({2, 3});
  Tensor b({3, 2});

  // 设置A的值：[[1,1,1], [2,2,2]]
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < 3; j++) {
      a.set({i, j}, (float)(i + 1));
    }
  }

  // 设置B的值：[[1,2], [1,2], [1,2]]
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 2; j++) {
      b.set({i, j}, (float)(j + 1));
    }
  }

  // 计算 C = A * B
  Tensor c = a.matmul(b);

  // 检查结果维度
  TEST_ASSERT(c.shape()[0] == 2, "结果第一维应为2");
  TEST_ASSERT(c.shape()[1] == 2, "结果第二维应为2");

  // 期望结果：[[3,6], [6,12]]
  TEST_ASSERT(c.get({0, 0}) == 3.0f, "C[0,0]应为3.0");
  TEST_ASSERT(c.get({0, 1}) == 6.0f, "C[0,1]应为6.0");
  TEST_ASSERT(c.get({1, 0}) == 6.0f, "C[1,0]应为6.0");
  TEST_ASSERT(c.get({1, 1}) == 12.0f, "C[1,1]应为12.0");

  return true;
}

// 测试激活函数
bool test_tensor_activations() {
  std::cout << "测试激活函数..." << std::endl;

  Tensor tensor({2, 2});
  tensor.set({0, 0}, -1.0f);
  tensor.set({0, 1}, 0.0f);
  tensor.set({1, 0}, 1.0f);
  tensor.set({1, 1}, 2.0f);

  // 测试ReLU
  Tensor relu_result = tensor.relu();
  TEST_ASSERT(relu_result.get({0, 0}) == 0.0f, "ReLU(-1) 应为 0");
  TEST_ASSERT(relu_result.get({0, 1}) == 0.0f, "ReLU(0) 应为 0");
  TEST_ASSERT(relu_result.get({1, 0}) == 1.0f, "ReLU(1) 应为 1");
  TEST_ASSERT(relu_result.get({1, 1}) == 2.0f, "ReLU(2) 应为 2");

  // 测试Sigmoid (近似值)
  Tensor sigmoid_result = tensor.sigmoid();
  float sigmoid_neg1 = 1.0f / (1.0f + std::exp(1.0f)); // ~0.269  使用std::exp
  float sigmoid_0 = 0.5f;                              // 1/(1+e^0) = 0.5
  float sigmoid_1 = 1.0f / (1.0f + std::exp(-1.0f));   // ~0.731
  float sigmoid_2 = 1.0f / (1.0f + std::exp(-2.0f));   // ~0.881

  TEST_ASSERT(std::abs(sigmoid_result.get({0, 0}) - sigmoid_neg1) < 0.001f,
              "Sigmoid(-1) ≈ 0.269");
  TEST_ASSERT(std::abs(sigmoid_result.get({0, 1}) - sigmoid_0) < 0.001f,
              "Sigmoid(0) = 0.5");
  TEST_ASSERT(std::abs(sigmoid_result.get({1, 0}) - sigmoid_1) < 0.001f,
              "Sigmoid(1) ≈ 0.731");
  TEST_ASSERT(std::abs(sigmoid_result.get({1, 1}) - sigmoid_2) < 0.001f,
              "Sigmoid(2) ≈ 0.881");

  return true;
}

// 基本张量创建和操作测试
TEST(TensorTest, Creation) {
  // 创建张量
  CNN::Tensor tensor({2, 3, 4});

  // 检查形状
  auto shape = tensor.shape();
  EXPECT_EQ(shape.size(), 3);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 3);
  EXPECT_EQ(shape[2], 4);

  // 检查元素数量
  EXPECT_EQ(tensor.size(), 24);

  // 检查维度数
  EXPECT_EQ(tensor.ndim(), 3);
}

// 填充测试
TEST(TensorTest, Fill) {
  CNN::Tensor tensor({2, 3});

  // 填充为常数
  tensor.fill(5.0f);

  // 检查所有元素是否为5
  for (size_t i = 0; i < tensor.size(); ++i) {
    EXPECT_FLOAT_EQ(tensor[i], 5.0f);
  }
}

// 重塑测试
TEST(TensorTest, Reshape) {
  CNN::Tensor tensor({2, 3, 4});

  // 重塑为6x4
  tensor.reshape({6, 4});

  // 检查新形状
  auto shape = tensor.shape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 6);
  EXPECT_EQ(shape[1], 4);

  // 元素数量应该不变
  EXPECT_EQ(tensor.size(), 24);
}

// 运算测试
TEST(TensorTest, Operations) {
  CNN::Tensor a({2, 2});
  CNN::Tensor b({2, 2});

  // 初始化
  a.fill(2.0f);
  b.fill(3.0f);

  // 加法
  CNN::Tensor c = a + b;
  for (size_t i = 0; i < c.size(); ++i) {
    EXPECT_FLOAT_EQ(c[i], 5.0f);
  }

  // 减法
  CNN::Tensor d = b - a;
  for (size_t i = 0; i < d.size(); ++i) {
    EXPECT_FLOAT_EQ(d[i], 1.0f);
  }

  // 乘法
  CNN::Tensor e = a * b;
  for (size_t i = 0; i < e.size(); ++i) {
    EXPECT_FLOAT_EQ(e[i], 6.0f);
  }

  // 标量乘法
  CNN::Tensor f = a * 3.0f;
  for (size_t i = 0; i < f.size(); ++i) {
    EXPECT_FLOAT_EQ(f[i], 6.0f);
  }
}

// 矩阵乘法测试
TEST(TensorTest, MatrixMultiplication) {
  CNN::Tensor a({2, 3});
  CNN::Tensor b({3, 2});

  // 初始化a为[[1,2,3],[4,5,6]]
  a[0] = 1.0f;
  a[1] = 2.0f;
  a[2] = 3.0f;
  a[3] = 4.0f;
  a[4] = 5.0f;
  a[5] = 6.0f;

  // 初始化b为[[7,8],[9,10],[11,12]]
  b[0] = 7.0f;
  b[1] = 8.0f;
  b[2] = 9.0f;
  b[3] = 10.0f;
  b[4] = 11.0f;
  b[5] = 12.0f;

  // 计算c = a*b
  CNN::Tensor c = a.matmul(b);

  // 检查c的形状应该是(2,2)
  auto shape = c.shape();
  EXPECT_EQ(shape.size(), 2);
  EXPECT_EQ(shape[0], 2);
  EXPECT_EQ(shape[1], 2);

  // 结果应该是[[58,64],[139,154]]
  EXPECT_FLOAT_EQ(c[0], 58.0f);
  EXPECT_FLOAT_EQ(c[1], 64.0f);
  EXPECT_FLOAT_EQ(c[2], 139.0f);
  EXPECT_FLOAT_EQ(c[3], 154.0f);
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}