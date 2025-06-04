#include "cnn/network.h"
#include "cnn/layers.h"
#include "cnn/loss.h"
#include "cnn/optimizer.h"
#include "cnn/utils.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace CNN {

Network::Network()
    : device_(Device::CPU), training_mode_(true), weight_decay_(0.0f),
      gradient_clip_norm_(0.0f), data_augmentation_(false), debug_mode_(false),
      early_stopping_enabled_(false), patience_counter_(0),
      best_val_loss_(INFINITY) {
  // 默认使用均方误差损失和SGD优化器
  loss_function_ = std::make_unique<MSELoss>();
  optimizer_ = std::make_unique<SGDOptimizer>(0.01f);
}

Network::~Network() = default;

void Network::add_layer(std::unique_ptr<Layer> layer) {
  layers_.push_back(std::move(layer));
}

void Network::clear_layers() { layers_.clear(); }

// 便捷的层添加方法
void Network::add_conv_layer(int out_channels, int kernel_size, int stride,
                             int padding, bool bias) {
  auto layer = std::make_unique<ConvLayer>(out_channels, kernel_size, stride,
                                           padding, bias);
  layers_.push_back(std::move(layer));
}

void Network::add_fc_layer(int out_features, bool bias) {
  auto layer = std::make_unique<FullyConnectedLayer>(out_features, bias);
  layers_.push_back(std::move(layer));
}

void Network::add_relu_layer() {
  auto layer = std::make_unique<ReLULayer>();
  layers_.push_back(std::move(layer));
}

void Network::add_sigmoid_layer() {
  auto layer = std::make_unique<SigmoidLayer>();
  layers_.push_back(std::move(layer));
}

void Network::add_tanh_layer() {
  auto layer = std::make_unique<TanhLayer>();
  layers_.push_back(std::move(layer));
}

void Network::add_softmax_layer(int dim) {
  auto layer = std::make_unique<SoftmaxLayer>(dim);
  layers_.push_back(std::move(layer));
}

void Network::add_maxpool_layer(int kernel_size, int stride, int padding) {
  if (stride == -1)
    stride = kernel_size;
  auto layer = std::make_unique<MaxPoolLayer>(kernel_size, stride, padding);
  layers_.push_back(std::move(layer));
}

void Network::add_avgpool_layer(int kernel_size, int stride, int padding) {
  if (stride == -1)
    stride = kernel_size;
  auto layer = std::make_unique<AvgPoolLayer>(kernel_size, stride, padding);
  layers_.push_back(std::move(layer));
}

void Network::add_dropout_layer(float p) {
  auto layer = std::make_unique<DropoutLayer>(p);
  layers_.push_back(std::move(layer));
}

void Network::add_batchnorm_layer(int num_features, float eps, float momentum) {
  auto layer = std::make_unique<BatchNormLayer>(num_features, eps, momentum);
  layers_.push_back(std::move(layer));
}

void Network::add_flatten_layer() {
  auto layer = std::make_unique<FlattenLayer>();
  layers_.push_back(std::move(layer));
}

// 前向传播
Tensor Network::forward(const Tensor &input) {
  Tensor output = input;
  for (auto &layer : layers_) {
    output = layer->forward(output);
  }
  return output;
}

Tensor Network::predict(const Tensor &input) {
  bool prev_mode = training_mode_;
  set_training_mode(false); // 设置为评估模式

  Tensor output = forward(input);

  // 恢复原来的模式
  set_training_mode(prev_mode);
  return output;
}

// 设置模式
void Network::set_training_mode(bool training) {
  training_mode_ = training;
  for (auto &layer : layers_) {
    layer->train(training);
  }
}

// 打印网络信息
void Network::print_summary(const std::vector<int> &input_shape) const {
  std::cout << "=== 网络架构摘要 ===" << std::endl;

  // 假设输入形状是有效的
  std::vector<size_t> current_shape(input_shape.begin(), input_shape.end());
  size_t total_params = 0;

  std::cout << "输入形状: [";
  for (size_t i = 0; i < current_shape.size(); ++i) {
    std::cout << current_shape[i];
    if (i < current_shape.size() - 1)
      std::cout << ", ";
  }
  std::cout << "]" << std::endl;

  for (size_t i = 0; i < layers_.size(); ++i) {
    const auto &layer = layers_[i];
    size_t layer_params = layer->parameters().size();
    total_params += layer_params;

    // 每个层的当前输出形状
    // 注意：这里需要每个层实现一个计算输出形状的方法
    std::cout << "第 " << i + 1 << " 层: " << layer->name()
              << ", 参数数量: " << layer_params << std::endl;
  }

  std::cout << "总参数数量: " << total_params << std::endl;
  std::cout << "===================" << std::endl;
}

// 辅助函数
std::vector<std::vector<Tensor>>
Network::create_batches(const std::vector<Tensor> &data,
                        const std::vector<Tensor> &labels, int batch_size) {
  // 检查数据和标签数量是否一致
  if (data.size() != labels.size()) {
    throw std::runtime_error("数据和标签数量不匹配");
  }

  // 创建索引并随机打乱
  std::vector<size_t> indices(data.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);

  // 将数据分成批次
  std::vector<std::vector<Tensor>> batches;
  int num_batches = (data.size() + batch_size - 1) / batch_size;

  for (int i = 0; i < num_batches; ++i) {
    std::vector<Tensor> batch;
    int start_idx = i * batch_size;
    int end_idx = std::min((i + 1) * batch_size, (int)data.size());

    for (int j = start_idx; j < end_idx; ++j) {
      batch.push_back(data[indices[j]]);
      batch.push_back(labels[indices[j]]);
    }

    batches.push_back(batch);
  }

  return batches;
}

// 设置优化器和损失函数
void Network::set_optimizer(std::unique_ptr<Optimizer> optimizer) {
  optimizer_ = std::move(optimizer);
}

void Network::set_loss_function(std::unique_ptr<LossFunction> loss_fn) {
  loss_function_ = std::move(loss_fn);
}

} // namespace CNN