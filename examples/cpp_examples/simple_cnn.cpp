#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

#include "cnn/layers.h"
#include "cnn/network.h"
#include "cnn/optimizer.h"
#include "cnn/tensor.h"
#include "cnn/utils.h"

// 生成简单的数据集
void generate_dataset(std::vector<CNN::Tensor> &inputs,
                      std::vector<CNN::Tensor> &targets, int num_samples,
                      int input_dim, int num_classes) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, 1.0f);

  inputs.resize(num_samples);
  targets.resize(num_samples);

  // 生成随机输入
  for (int i = 0; i < num_samples; i++) {
    inputs[i] = CNN::Tensor({1, (size_t)input_dim, (size_t)input_dim});
    inputs[i].rand(0.0f, 0.1f, i);

    // 随机生成分类
    int target_class = i % num_classes;
    targets[i] = CNN::Tensor({(size_t)num_classes});
    targets[i].fill(0.0f);
    targets[i][target_class] = 1.0f;
  }
}

int main() {
  std::cout << "===== 简单CNN演示 =====\n\n";

  // 参数
  const int input_dim = 28;   // 输入图像尺寸 (28x28)
  const int num_classes = 10; // 类别数量
  const int num_train = 100;  // 训练样本数量（减少以便演示）
  const int num_test = 20;    // 测试样本数量（减少以便演示）

  // 生成训练数据
  std::cout << "生成训练数据和测试数据...\n";
  std::vector<CNN::Tensor> train_inputs, train_targets;
  std::vector<CNN::Tensor> test_inputs, test_targets;

  generate_dataset(train_inputs, train_targets, num_train, input_dim,
                   num_classes);
  generate_dataset(test_inputs, test_targets, num_test, input_dim, num_classes);

  // 创建网络
  std::cout << "创建CNN网络...\n";
  CNN::Network network;

  // 添加网络层
  network.add_conv_layer(16, 3, 1, 1); // 卷积层: 16个3x3卷积核，步长1，填充1
  network.add_relu_layer();            // ReLU激活
  network.add_maxpool_layer(2, 2);     // 最大池化: 2x2, 步长2
  network.add_conv_layer(32, 3, 1, 1); // 卷积层: 32个3x3卷积核，步长1，填充1
  network.add_relu_layer();            // ReLU激活
  network.add_maxpool_layer(2, 2);     // 最大池化: 2x2, 步长2
  network.add_flatten_layer();         // 展平层
  network.add_fc_layer(128);           // 全连接层: 输出128
  network.add_relu_layer();            // ReLU激活
  network.add_fc_layer(num_classes);   // 全连接层: 输出类别数
  network.add_softmax_layer();         // Softmax激活

  // 设置优化器和损失函数
  network.set_optimizer(std::make_unique<CNN::AdamOptimizer>(0.001f));
  network.set_loss_function(std::make_unique<CNN::CrossEntropyLoss>());

  // 打印网络结构
  std::cout << "\n网络结构:\n";
  network.print_summary({1, input_dim, input_dim});

  // 简化的演示过程
  std::cout << "\n开始简化演示...\n";
  auto start_time = std::chrono::high_resolution_clock::now();

  // 演示前向传播
  for (int epoch = 1; epoch <= 3; ++epoch) {
    std::cout << "轮次 " << epoch << "/3 - 演示前向传播..." << std::endl;

    // 随机选择几个样本进行前向传播演示
    for (int i = 0; i < 5 && i < train_inputs.size(); ++i) {
      CNN::Tensor output = network.forward(train_inputs[i]);

      // 找到最大值索引
      float max_val = output[0];
      int predicted = 0;
      for (int j = 1; j < num_classes; j++) {
        if (output[j] > max_val) {
          max_val = output[j];
          predicted = j;
        }
      }

      // 找到实际类别
      int actual = 0;
      for (int j = 0; j < num_classes; j++) {
        if (train_targets[i][j] > 0.5f) {
          actual = j;
          break;
        }
      }

      std::cout << "样本 " << i + 1 << ": 预测类别=" << predicted
                << ", 实际类别=" << actual << std::endl;
    }
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
      end_time - start_time);

  std::cout << "\n演示完成！耗时: " << duration.count() / 1000.0f << " 秒\n";

  // 简化的测试集演示
  std::cout << "\n在测试集上演示预测...\n";
  int correct = 0;

  for (int i = 0; i < std::min(10, (int)test_inputs.size()); i++) {
    CNN::Tensor output = network.predict(test_inputs[i]);

    // 找到最大值索引
    float max_val = output[0];
    int predicted = 0;
    for (int j = 1; j < num_classes; j++) {
      if (output[j] > max_val) {
        max_val = output[j];
        predicted = j;
      }
    }

    // 找到实际类别
    int actual = 0;
    for (int j = 0; j < num_classes; j++) {
      if (test_targets[i][j] > 0.5f) {
        actual = j;
        break;
      }
    }

    if (predicted == actual) {
      correct++;
    }

    std::cout << "样本 " << i << ": 预测类别 = " << predicted
              << ", 实际类别 = " << actual << ", 置信度 = " << max_val
              << std::endl;
  }

  float accuracy = (float)correct / std::min(10, (int)test_inputs.size());
  std::cout << "演示准确率: " << (accuracy * 100.0f) << "%\n";

  std::cout << "\n演示完成！\n";
  return 0;
}