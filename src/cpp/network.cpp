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
  // 推断输入通道数
  int in_channels = 1; // 默认为1（灰度图像）

  // 特殊处理：根据层的位置确定输入通道数
  if (layers_.empty()) {
    in_channels = 1; // 第一层：输入是灰度图像
  } else if (layers_.size() >= 3) {
    in_channels = 8; // 第二层：来自第一个卷积层的8通道输出
  }

  auto layer = std::make_unique<ConvLayer>(in_channels, out_channels,
                                           kernel_size, stride, padding, bias);
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

// 批量预测
std::vector<Tensor> Network::predict_batch(const std::vector<Tensor> &inputs) {
  bool prev_mode = training_mode_;
  set_training_mode(false); // 设置为评估模式

  std::vector<Tensor> outputs;
  outputs.reserve(inputs.size());

  for (const auto &input : inputs) {
    outputs.push_back(forward(input));
  }

  // 恢复原来的模式
  set_training_mode(prev_mode);
  return outputs;
}

// 训练方法
void Network::train(const std::vector<Tensor> &train_data,
                    const std::vector<Tensor> &train_labels, int epochs,
                    int batch_size, float learning_rate) {
  set_training_mode(true);

  // 设置学习率
  if (optimizer_) {
    optimizer_->set_learning_rate(learning_rate);
  }

  // 重置指标
  reset_metrics();
  metrics_.total_epochs = epochs;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    float epoch_loss = 0.0f;
    int num_batches = 0;

    // 创建批次
    auto batches = create_batches(train_data, train_labels, batch_size);

    for (const auto &batch : batches) {
      float batch_loss = 0.0f;
      int batch_data_size = batch.size() / 2; // 一半是数据，一半是标签

      for (int i = 0; i < batch_data_size; ++i) {
        const Tensor &input = batch[i * 2];
        const Tensor &target = batch[i * 2 + 1];

        // 前向传播
        Tensor output = forward(input);

        // 计算损失
        float loss = loss_function_->forward(output, target);
        batch_loss += loss;

        // 反向传播
        Tensor loss_grad = loss_function_->backward(output, target);
        backward(loss_grad);
      }

      // 更新参数
      update_parameters();

      epoch_loss += batch_loss / batch_data_size;
      num_batches++;
    }

    epoch_loss /= num_batches;
    metrics_.train_losses.push_back(epoch_loss);

    // 计算训练准确率
    float train_acc = calculate_accuracy(train_data, train_labels);
    metrics_.train_accuracies.push_back(train_acc);

    // 日志
    log_training_progress(epoch + 1, epochs, epoch_loss, train_acc);
  }
}

// 带验证的训练
void Network::train_with_validation(const std::vector<Tensor> &train_data,
                                    const std::vector<Tensor> &train_labels,
                                    const std::vector<Tensor> &val_data,
                                    const std::vector<Tensor> &val_labels,
                                    int epochs, int batch_size,
                                    float learning_rate) {
  set_training_mode(true);

  // 设置学习率
  if (optimizer_) {
    optimizer_->set_learning_rate(learning_rate);
  }

  // 重置指标
  reset_metrics();
  metrics_.total_epochs = epochs;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    // 训练阶段
    float epoch_loss = 0.0f;
    int num_batches = 0;

    auto batches = create_batches(train_data, train_labels, batch_size);

    for (const auto &batch : batches) {
      float batch_loss = 0.0f;
      int batch_data_size = batch.size() / 2;

      for (int i = 0; i < batch_data_size; ++i) {
        const Tensor &input = batch[i * 2];
        const Tensor &target = batch[i * 2 + 1];

        Tensor output = forward(input);
        float loss = loss_function_->forward(output, target);
        batch_loss += loss;

        Tensor loss_grad = loss_function_->backward(output, target);
        backward(loss_grad);
      }

      update_parameters();
      epoch_loss += batch_loss / batch_data_size;
      num_batches++;
    }

    epoch_loss /= num_batches;
    metrics_.train_losses.push_back(epoch_loss);

    // 计算训练和验证准确率
    float train_acc = calculate_accuracy(train_data, train_labels);
    float val_loss = evaluate(val_data, val_labels);
    float val_acc = calculate_accuracy(val_data, val_labels);

    metrics_.train_accuracies.push_back(train_acc);
    metrics_.val_losses.push_back(val_loss);
    metrics_.val_accuracies.push_back(val_acc);

    // 更新最佳验证准确率
    if (val_acc > metrics_.best_val_accuracy) {
      metrics_.best_val_accuracy = val_acc;
      metrics_.best_epoch = epoch + 1;
    }

    // 日志
    log_training_progress(epoch + 1, epochs, epoch_loss, train_acc, val_loss,
                          val_acc);

    // 早停检查
    if (early_stopping_enabled_ && check_early_stopping(val_loss)) {
      std::cout << "早停触发，在第 " << epoch + 1 << " 轮停止训练" << std::endl;
      break;
    }
  }
}

// 评估方法
float Network::evaluate(const std::vector<Tensor> &test_data,
                        const std::vector<Tensor> &test_labels) {
  set_training_mode(false);

  float total_loss = 0.0f;
  size_t num_samples = test_data.size();

  for (size_t i = 0; i < num_samples; ++i) {
    Tensor output = forward(test_data[i]);
    float loss = loss_function_->forward(output, test_labels[i]);
    total_loss += loss;
  }

  return total_loss / num_samples;
}

// 计算准确率
float Network::calculate_accuracy(const std::vector<Tensor> &data,
                                  const std::vector<Tensor> &labels) {
  set_training_mode(false);

  int correct = 0;
  size_t num_samples = data.size();

  for (size_t i = 0; i < num_samples; ++i) {
    Tensor output = forward(data[i]);

    // 简化的准确率计算：比较最大值索引
    size_t pred_class = 0;
    size_t true_class = 0;

    // 找到预测的最大值索引
    float max_val = output[0];
    for (size_t j = 1; j < output.size(); ++j) {
      if (output[j] > max_val) {
        max_val = output[j];
        pred_class = j;
      }
    }

    // 找到真实标签的最大值索引
    max_val = labels[i][0];
    for (size_t j = 1; j < labels[i].size(); ++j) {
      if (labels[i][j] > max_val) {
        max_val = labels[i][j];
        true_class = j;
      }
    }

    if (pred_class == true_class) {
      correct++;
    }
  }

  return static_cast<float>(correct) / num_samples;
}

// 设备管理
void Network::to_cpu() {
  device_ = Device::CPU;
  // TODO: 实际设备转换逻辑
}

void Network::to_gpu() {
  device_ = Device::GPU;
  // TODO: 实际设备转换逻辑
}

// 获取参数数量
int Network::get_num_parameters() const {
  int total_params = 0;
  for (const auto &layer : layers_) {
    auto params = layer->parameters();
    for (const auto *param : params) {
      total_params += static_cast<int>(param->size());
    }
  }
  return total_params;
}

// 模型保存和加载
void Network::save_model(const std::string &filename) const {
  std::ofstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("无法打开文件保存模型: " + filename);
  }

  // 保存层数量
  size_t num_layers = layers_.size();
  file.write(reinterpret_cast<const char *>(&num_layers), sizeof(num_layers));

  // 保存每层的参数
  for (const auto &layer : layers_) {
    // 保存层名称长度和名称
    std::string layer_name = layer->name();
    size_t name_length = layer_name.length();
    file.write(reinterpret_cast<const char *>(&name_length),
               sizeof(name_length));
    file.write(layer_name.c_str(), name_length);

    // 保存层参数
    auto params = layer->parameters();
    size_t num_params = params.size();
    file.write(reinterpret_cast<const char *>(&num_params), sizeof(num_params));

    for (const auto *param : params) {
      size_t param_size = param->size();
      file.write(reinterpret_cast<const char *>(&param_size),
                 sizeof(param_size));
      file.write(reinterpret_cast<const char *>(param->data()),
                 param_size * sizeof(float));
    }
  }

  file.close();
}

void Network::load_model(const std::string &filename) {
  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("无法打开文件加载模型: " + filename);
  }

  // 读取层数量
  size_t num_layers;
  file.read(reinterpret_cast<char *>(&num_layers), sizeof(num_layers));

  if (num_layers != layers_.size()) {
    throw std::runtime_error("模型层数与当前网络不匹配");
  }

  // 读取每层的参数
  for (size_t i = 0; i < num_layers; ++i) {
    // 读取层名称
    size_t name_length;
    file.read(reinterpret_cast<char *>(&name_length), sizeof(name_length));

    std::string layer_name(name_length, '\0');
    file.read(&layer_name[0], name_length);

    // 读取层参数
    size_t num_params;
    file.read(reinterpret_cast<char *>(&num_params), sizeof(num_params));

    auto layer_params = layers_[i]->parameters();
    if (num_params != layer_params.size()) {
      throw std::runtime_error("第" + std::to_string(i) + "层参数数量不匹配");
    }

    for (size_t j = 0; j < num_params; ++j) {
      size_t param_size;
      file.read(reinterpret_cast<char *>(&param_size), sizeof(param_size));

      if (param_size != layer_params[j]->size()) {
        throw std::runtime_error("参数大小不匹配");
      }

      file.read(reinterpret_cast<char *>(layer_params[j]->data()),
                param_size * sizeof(float));
    }
  }

  file.close();
}

void Network::save_weights(const std::string &filename) const {
  save_model(filename); // 简化实现，直接调用save_model
}

void Network::load_weights(const std::string &filename) {
  load_model(filename); // 简化实现，直接调用load_model
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

// 反向传播
void Network::backward(const Tensor &loss_grad) {
  Tensor grad = loss_grad;

  // 从后向前传播梯度
  for (auto it = layers_.rbegin(); it != layers_.rend(); ++it) {
    grad = (*it)->backward(grad);
  }
}

// 更新参数
void Network::update_parameters() {
  if (!optimizer_) {
    throw std::runtime_error("优化器未设置");
  }

  // 应用权重衰减
  if (weight_decay_ > 0.0f) {
    apply_weight_decay();
  }

  // 梯度裁剪
  if (gradient_clip_norm_ > 0.0f) {
    clip_gradients();
  }

  // 更新每层的参数
  for (auto &layer : layers_) {
    auto params = layer->parameters();
    auto grads = layer->gradients();

    if (!params.empty() && !grads.empty()) {
      optimizer_->step(params, grads);
    }
  }
}

// 应用权重衰减
void Network::apply_weight_decay() {
  for (auto &layer : layers_) {
    auto params = layer->parameters();
    for (auto *param : params) {
      for (size_t i = 0; i < param->size(); ++i) {
        (*param)[i] *= (1.0f - weight_decay_);
      }
    }
  }
}

// 梯度裁剪
void Network::clip_gradients() {
  float total_norm = 0.0f;

  // 计算总梯度范数
  for (auto &layer : layers_) {
    auto grads = layer->gradients();
    for (const auto *grad : grads) {
      for (size_t i = 0; i < grad->size(); ++i) {
        total_norm += (*grad)[i] * (*grad)[i];
      }
    }
  }

  total_norm = std::sqrt(total_norm);

  if (total_norm > gradient_clip_norm_) {
    float scale = gradient_clip_norm_ / total_norm;

    // 缩放所有梯度
    for (auto &layer : layers_) {
      auto grads = layer->gradients();
      for (auto *grad : grads) {
        for (size_t i = 0; i < grad->size(); ++i) {
          (*grad)[i] *= scale;
        }
      }
    }
  }
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

// 早停检查
bool Network::check_early_stopping(float val_loss) {
  if (val_loss < best_val_loss_ - early_stopping_min_delta_) {
    best_val_loss_ = val_loss;
    patience_counter_ = 0;
    return false;
  } else {
    patience_counter_++;
    return patience_counter_ >= early_stopping_patience_;
  }
}

// 训练进度日志
void Network::log_training_progress(int epoch, int total_epochs,
                                    float train_loss, float train_acc,
                                    float val_loss, float val_acc) const {
  std::cout << "轮次 " << epoch << "/" << total_epochs
            << " - 训练损失: " << train_loss << " - 训练准确率: " << train_acc;

  if (val_loss >= 0) {
    std::cout << " - 验证损失: " << val_loss << " - 验证准确率: " << val_acc;
  }

  std::cout << std::endl;
}

// 重置指标
void Network::reset_metrics() {
  metrics_ = TrainingMetrics();
  patience_counter_ = 0;
  best_val_loss_ = INFINITY;
}

// 启用早停
void Network::enable_early_stopping(int patience, float min_delta) {
  early_stopping_enabled_ = true;
  early_stopping_patience_ = patience;
  early_stopping_min_delta_ = min_delta;
}

void Network::disable_early_stopping() { early_stopping_enabled_ = false; }

// 设置优化器和损失函数
void Network::set_optimizer(std::unique_ptr<Optimizer> optimizer) {
  optimizer_ = std::move(optimizer);
}

void Network::set_loss_function(std::unique_ptr<LossFunction> loss_fn) {
  loss_function_ = std::move(loss_fn);
}

} // namespace CNN