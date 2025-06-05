#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "cnn/layers.h"
#include "cnn/network.h"
#include "cnn/optimizer.h"
#include "cnn/tensor.h"
#include "cnn/utils.h"

// 智能获取项目数据目录路径
std::string get_project_data_path() {
  // 1. 首先尝试从环境变量获取项目根目录
  const char *project_root = std::getenv("CNN_DEMO_ROOT");
  if (project_root) {
    std::string data_path = std::string(project_root) + "/data";
    if (std::filesystem::exists(data_path)) {
      std::cout << "使用环境变量CNN_DEMO_ROOT指定的数据路径: " << data_path
                << std::endl;
      return data_path;
    }
  }

  // 2. 尝试相对于当前工作目录的常见路径
  std::vector<std::string> possible_paths = {
      "./data",          // 从项目根目录运行
      "../data",         // 从build目录运行
      "../../data",      // 从build/bin目录运行
      "../../../data",   // 从build/Debug/bin等深层目录运行
      "./CNN_demo/data", // 从上级目录运行
      "../CNN_demo/data" // 从兄弟目录运行
  };

  for (const auto &path : possible_paths) {
    if (std::filesystem::exists(path)) {
      std::cout << "找到数据目录: " << std::filesystem::absolute(path)
                << std::endl;
      return path;
    }
  }

  // 3. 如果都找不到，返回默认路径（会触发生成随机数据）
  std::cout << "未找到数据目录，将使用随机数据进行演示" << std::endl;
  return "./data"; // 默认路径，即使不存在也没关系，load_mnist_sample会处理
}

// 简单的MNIST数据加载函数（部分数据用于演示）
bool load_mnist_sample(std::vector<CNN::Tensor> &images,
                       std::vector<int> &labels, const std::string &images_path,
                       const std::string &labels_path, int max_samples = 1000) {
  std::cout << "加载MNIST样本数据...\n";

  // 如果文件不存在，生成随机数据用于演示
  std::ifstream img_file(images_path, std::ios::binary);
  std::ifstream lbl_file(labels_path, std::ios::binary);

  if (!img_file.is_open() || !lbl_file.is_open()) {
    std::cout << "未找到MNIST数据文件，生成随机数据用于演示...\n";

    // 随机数生成器
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> label_dist(0, 9);

    // 生成随机图像和标签
    for (int i = 0; i < max_samples; ++i) {
      // 创建28x28的随机图像
      CNN::Tensor img({1, 28, 28});
      for (size_t j = 0; j < img.size(); ++j) {
        img.data()[j] = dist(gen);
      }
      images.push_back(img);

      // 随机标签 (0-9)
      labels.push_back(label_dist(gen));
    }

    std::cout << "已生成 " << max_samples << " 个随机样本\n";
    return true;
  }

  // 读取MNIST文件头
  uint32_t magic, num_images, rows, cols;
  uint32_t labels_magic, num_labels;

  img_file.read(reinterpret_cast<char *>(&magic), 4);
  img_file.read(reinterpret_cast<char *>(&num_images), 4);
  img_file.read(reinterpret_cast<char *>(&rows), 4);
  img_file.read(reinterpret_cast<char *>(&cols), 4);

  lbl_file.read(reinterpret_cast<char *>(&labels_magic), 4);
  lbl_file.read(reinterpret_cast<char *>(&num_labels), 4);

  // 大端字节序转换
  magic = ((magic & 0xff000000) >> 24) | ((magic & 0x00ff0000) >> 8) |
          ((magic & 0x0000ff00) << 8) | ((magic & 0x000000ff) << 24);
  num_images =
      ((num_images & 0xff000000) >> 24) | ((num_images & 0x00ff0000) >> 8) |
      ((num_images & 0x0000ff00) << 8) | ((num_images & 0x000000ff) << 24);
  rows = ((rows & 0xff000000) >> 24) | ((rows & 0x00ff0000) >> 8) |
         ((rows & 0x0000ff00) << 8) | ((rows & 0x000000ff) << 24);
  cols = ((cols & 0xff000000) >> 24) | ((cols & 0x00ff0000) >> 8) |
         ((cols & 0x0000ff00) << 8) | ((cols & 0x000000ff) << 24);

  labels_magic =
      ((labels_magic & 0xff000000) >> 24) | ((labels_magic & 0x00ff0000) >> 8) |
      ((labels_magic & 0x0000ff00) << 8) | ((labels_magic & 0x000000ff) << 24);
  num_labels =
      ((num_labels & 0xff000000) >> 24) | ((num_labels & 0x00ff0000) >> 8) |
      ((num_labels & 0x0000ff00) << 8) | ((num_labels & 0x000000ff) << 24);

  // 检查文件格式
  if (magic != 0x803 || labels_magic != 0x801) {
    std::cerr << "无效的MNIST文件格式\n";
    return false;
  }

  // 限制样本数量
  int count = std::min(static_cast<int>(num_images), max_samples);

  // 读取图像和标签
  std::vector<unsigned char> buffer(rows * cols);
  unsigned char label;

  for (int i = 0; i < count; ++i) {
    // 读取一张图像
    img_file.read(reinterpret_cast<char *>(buffer.data()), rows * cols);

    // 将图像数据转换为Tensor (归一化到0-1)
    CNN::Tensor img({1, rows, cols});
    for (size_t j = 0; j < buffer.size(); ++j) {
      img.data()[j] = static_cast<float>(buffer[j]) / 255.0f;
    }
    images.push_back(img);

    // 读取对应的标签
    lbl_file.read(reinterpret_cast<char *>(&label), 1);
    labels.push_back(static_cast<int>(label));
  }

  std::cout << "已加载 " << count << " 个MNIST样本\n";
  return true;
}

// 将整数标签转换为one-hot编码
CNN::Tensor to_one_hot(int label, int num_classes = 10) {
  CNN::Tensor one_hot({(size_t)num_classes});
  std::fill(one_hot.data(), one_hot.data() + one_hot.size(), 0.0f);
  one_hot.data()[label] = 1.0f;
  return one_hot;
}

// 评估模型性能
void evaluate_model(CNN::Network &network,
                    const std::vector<CNN::Tensor> &test_images,
                    const std::vector<int> &test_labels) {
  int correct = 0;
  float total_loss = 0.0f;

  network.eval_mode(); // 设置为评估模式

  for (size_t i = 0; i < test_images.size(); ++i) {
    // 前向传播
    CNN::Tensor output = network.predict(test_images[i]);

    // 计算预测类别
    int predicted = 0;
    float max_val = output.data()[0];
    for (int j = 1; j < 10; ++j) {
      if (output.data()[j] > max_val) {
        max_val = output.data()[j];
        predicted = j;
      }
    }

    // 计算准确率
    if (predicted == test_labels[i]) {
      correct++;
    }

    // 简化：不计算损失，因为没有相应的方法
  }

  float accuracy = static_cast<float>(correct) / test_images.size();

  std::cout << "测试结果 - 准确率: " << accuracy * 100.0f << "%" << std::endl;
}

// 保存训练历史到CSV文件
void save_history_csv(const std::vector<float> &train_losses,
                      const std::vector<float> &train_accuracies,
                      const std::string &filename) {
  std::ofstream file(filename);
  if (!file.is_open()) {
    std::cerr << "无法创建文件: " << filename << std::endl;
    return;
  }

  file << "epoch,loss,accuracy\n";
  for (size_t i = 0; i < train_losses.size(); ++i) {
    file << i + 1 << "," << train_losses[i] << "," << train_accuracies[i]
         << "\n";
  }

  file.close();
  std::cout << "训练历史已保存到: " << filename << std::endl;
}

int main() {
  std::cout << "=== CNN混合架构演示: MNIST训练 ===\n\n";

  // 加载MNIST数据（或生成随机数据）
  std::vector<CNN::Tensor> train_images;
  std::vector<int> train_labels;
  std::vector<CNN::Tensor> test_images;
  std::vector<int> test_labels;

  // 智能获取数据文件路径
  std::string data_dir = get_project_data_path();
  std::string train_images_file = data_dir + "/train-images.idx3-ubyte";
  std::string train_labels_file = data_dir + "/train-labels.idx1-ubyte";
  std::string test_images_file = data_dir + "/t10k-images.idx3-ubyte";
  std::string test_labels_file = data_dir + "/t10k-labels.idx1-ubyte";

  // 加载训练数据和测试数据（如果不存在，将生成随机数据）
  load_mnist_sample(train_images, train_labels, train_images_file,
                    train_labels_file, 8000); // 增加训练样本到8000
  load_mnist_sample(test_images, test_labels, test_images_file,
                    test_labels_file, 2000); // 增加测试样本到2000

  // 创建CNN网络
  CNN::Network network;

  // 构建最优的网络架构
  network.add_conv_layer(8, 5, 1, 2); // 第一层：1→8通道
  network.add_relu_layer();
  network.add_maxpool_layer(2, 2);     // 28x28 → 14x14
  network.add_conv_layer(16, 5, 1, 0); // 第二层：8→16通道
  network.add_relu_layer();
  network.add_maxpool_layer(2, 2); // 14x14 → 7x7
  network.add_flatten_layer();     // 展平
  network.add_fc_layer(128);       // 第一个全连接层
  network.add_relu_layer();
  network.add_dropout_layer(0.4f); // 增加Dropout到40%
  network.add_fc_layer(64);        // 第二个全连接层
  network.add_relu_layer();
  network.add_dropout_layer(0.3f); // Dropout 30%
  network.add_fc_layer(10);        // 输出层10个类别

  // 设置优化器和损失函数
  network.set_optimizer(
      std::make_unique<CNN::SGDOptimizer>(0.02f)); // 稍微降低学习率
  network.set_loss_function(
      std::make_unique<CNN::CrossEntropyLoss>(true)); // from_logits=true

  // 转换标签为one-hot编码
  std::vector<CNN::Tensor> train_one_hot_labels;
  std::vector<CNN::Tensor> test_one_hot_labels;

  for (int label : train_labels) {
    train_one_hot_labels.push_back(to_one_hot(label));
  }
  for (int label : test_labels) {
    test_one_hot_labels.push_back(to_one_hot(label));
  }

  std::cout << "网络参数数量: " << network.get_num_parameters() << std::endl;

  // 触发参数初始化：进行一次前向传播
  if (!train_images.empty()) {
    std::cout << "触发参数初始化..." << std::endl;
    CNN::Tensor dummy_output = network.forward(train_images[0]);
    std::cout << "初始化后网络参数数量: " << network.get_num_parameters()
              << std::endl;
  }

  // 开始训练
  std::cout << "\n开始训练...\n";

  // 使用Network类的train方法进行训练，最终调优参数
  network.train(train_images, train_one_hot_labels, 20, 32,
                0.02f); // 增加到20轮

  // 评估模型
  std::cout << "\n在测试集上评估模型...\n";
  evaluate_model(network, test_images, test_labels);

  std::cout << "\n训练完成!\n";
  return 0;
}