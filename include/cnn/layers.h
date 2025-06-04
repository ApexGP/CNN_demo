#pragma once

#include "tensor.h"
#include <memory>
#include <string>

namespace CNN {

// 基础层抽象类
class Layer {
public:
  virtual ~Layer() = default;

  // 前向传播
  virtual Tensor forward(const Tensor &input) = 0;

  // 反向传播
  virtual Tensor backward(const Tensor &grad_output) = 0;

  // 获取参数
  virtual std::vector<Tensor *> parameters() { return {}; }

  // 获取梯度
  virtual std::vector<Tensor *> gradients() { return {}; }

  // 设置训练模式
  virtual void train(bool mode = true) { training_ = mode; }
  virtual bool is_training() const { return training_; }

  // 层信息
  virtual std::string name() const = 0;
  virtual std::vector<int>
  output_shape(const std::vector<int> &input_shape) const = 0;

protected:
  bool training_ = true;
};

// 卷积层
class ConvLayer : public Layer {
public:
  ConvLayer(int out_channels, int kernel_size, int stride = 1, int padding = 0,
            bool bias = true);
  ConvLayer(int in_channels, int out_channels, int kernel_size, int stride = 1,
            int padding = 0, bool bias = true);

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;

  std::string name() const override { return "ConvLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

  // 卷积层特有方法
  void set_padding(int padding) { padding_ = padding; }
  void set_stride(int stride) { stride_ = stride; }

private:
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;
  bool use_bias_;

  Tensor weights_;
  Tensor bias_;
  Tensor weight_grad_;
  Tensor bias_grad_;

  // 用于反向传播的中间结果
  Tensor last_input_;

  void initialize_parameters();
};

// 全连接层
class FullyConnectedLayer : public Layer {
public:
  FullyConnectedLayer(int out_features, bool bias = true);
  FullyConnectedLayer(int in_features, int out_features, bool bias = true);

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;

  std::string name() const override { return "FullyConnectedLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  int in_features_;
  int out_features_;
  bool use_bias_;

  Tensor weights_;
  Tensor bias_;
  Tensor weight_grad_;
  Tensor bias_grad_;

  Tensor last_input_;

  void initialize_parameters();
};

// ReLU激活层
class ReLULayer : public Layer {
public:
  ReLULayer() = default;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::string name() const override { return "ReLULayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  Tensor last_input_;
};

// Sigmoid激活层
class SigmoidLayer : public Layer {
public:
  SigmoidLayer() = default;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::string name() const override { return "SigmoidLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  Tensor last_output_;
};

// Tanh激活层
class TanhLayer : public Layer {
public:
  TanhLayer() = default;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::string name() const override { return "TanhLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  Tensor last_output_;
};

// Softmax层
class SoftmaxLayer : public Layer {
public:
  SoftmaxLayer(int dim = -1) : dim_(dim) {}

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::string name() const override { return "SoftmaxLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  int dim_;
  Tensor last_output_;
};

// 最大池化层
class MaxPoolLayer : public Layer {
public:
  MaxPoolLayer(int kernel_size, int stride = -1, int padding = 0);

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::string name() const override { return "MaxPoolLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  int kernel_size_;
  int stride_;
  int padding_;

  Tensor last_input_;
  Tensor max_indices_;
};

// 平均池化层
class AvgPoolLayer : public Layer {
public:
  AvgPoolLayer(int kernel_size, int stride = -1, int padding = 0);

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::string name() const override { return "AvgPoolLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  int kernel_size_;
  int stride_;
  int padding_;

  Tensor last_input_;
};

// Dropout层
class DropoutLayer : public Layer {
public:
  DropoutLayer(float p = 0.5) : p_(p) {}

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::string name() const override { return "DropoutLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  float p_;
  Tensor dropout_mask_;
};

// 批标准化层
class BatchNormLayer : public Layer {
public:
  BatchNormLayer(int num_features, float eps = 1e-5, float momentum = 0.1);

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::vector<Tensor *> parameters() override;
  std::vector<Tensor *> gradients() override;

  std::string name() const override { return "BatchNormLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  int num_features_;
  float eps_;
  float momentum_;

  Tensor gamma_; // 缩放参数
  Tensor beta_;  // 偏移参数
  Tensor gamma_grad_;
  Tensor beta_grad_;

  // 运行时统计
  Tensor running_mean_;
  Tensor running_var_;

  // 用于反向传播
  Tensor last_input_;
  Tensor normalized_;
  Tensor std_;

  void initialize_parameters();
};

// Flatten层
class FlattenLayer : public Layer {
public:
  FlattenLayer() = default;

  Tensor forward(const Tensor &input) override;
  Tensor backward(const Tensor &grad_output) override;

  std::string name() const override { return "FlattenLayer"; }
  std::vector<int>
  output_shape(const std::vector<int> &input_shape) const override;

private:
  std::vector<int> last_input_shape_;
};

} // namespace CNN