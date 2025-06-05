#include "cnn/layers.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace CNN {

// ConvLayer实现
ConvLayer::ConvLayer(int out_channels, int kernel_size, int stride, int padding,
                     bool bias)
    : out_channels_(out_channels), kernel_size_(kernel_size), stride_(stride),
      padding_(padding), use_bias_(bias), in_channels_(0) {}

ConvLayer::ConvLayer(int in_channels, int out_channels, int kernel_size,
                     int stride, int padding, bool bias)
    : in_channels_(in_channels), out_channels_(out_channels),
      kernel_size_(kernel_size), stride_(stride), padding_(padding),
      use_bias_(bias) {
  initialize_parameters();
}

void ConvLayer::initialize_parameters() {
  if (in_channels_ > 0) {
    weights_ = Tensor({(size_t)out_channels_, (size_t)in_channels_,
                       (size_t)kernel_size_, (size_t)kernel_size_});
    // Xavier初始化
    float fan_in = in_channels_ * kernel_size_ * kernel_size_;
    float fan_out = out_channels_ * kernel_size_ * kernel_size_;
    weights_.xavier_uniform(fan_in, fan_out);

    if (use_bias_) {
      bias_ = Tensor({(size_t)out_channels_});
      bias_.zeros();
    }
    weight_grad_ = Tensor(weights_.shape());
    weight_grad_.zeros();
    if (use_bias_) {
      bias_grad_ = Tensor(bias_.shape());
      bias_grad_.zeros();
    }
  }
}

Tensor ConvLayer::forward(const Tensor &input) {
  last_input_ = input;

  // 如果还没有初始化参数，根据输入形状初始化
  if (in_channels_ == 0 && input.ndim() >= 3) {
    in_channels_ = input.shape()[input.ndim() - 3];
    initialize_parameters();
  }

  // 简化的卷积实现（假设输入是CHW格式）
  if (input.ndim() != 3) {
    throw std::runtime_error("ConvLayer期望3D输入 (C, H, W)");
  }

  size_t in_c = input.shape()[0];
  size_t in_h = input.shape()[1];
  size_t in_w = input.shape()[2];

  size_t out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
  size_t out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;

  Tensor output({(size_t)out_channels_, out_h, out_w});
  output.zeros();

  // 简化的卷积计算（无优化版本）
  for (int oc = 0; oc < out_channels_; oc++) {
    for (size_t oh = 0; oh < out_h; oh++) {
      for (size_t ow = 0; ow < out_w; ow++) {
        float sum = 0.0f;

        for (int ic = 0; ic < in_channels_; ic++) {
          for (int kh = 0; kh < kernel_size_; kh++) {
            for (int kw = 0; kw < kernel_size_; kw++) {
              int ih = oh * stride_ - padding_ + kh;
              int iw = ow * stride_ - padding_ + kw;

              if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                size_t input_idx = ic * in_h * in_w + ih * in_w + iw;
                size_t weight_idx =
                    oc * in_channels_ * kernel_size_ * kernel_size_ +
                    ic * kernel_size_ * kernel_size_ + kh * kernel_size_ + kw;
                sum += input[input_idx] * weights_[weight_idx];
              }
            }
          }
        }

        if (use_bias_) {
          sum += bias_[oc];
        }

        size_t output_idx = oc * out_h * out_w + oh * out_w + ow;
        output.data()[output_idx] = sum;
      }
    }
  }

  return output;
}

Tensor ConvLayer::backward(const Tensor &grad_output) {
  if (last_input_.size() == 0 || in_channels_ == 0) {
    return grad_output.clone();
  }

  // 获取形状信息
  size_t in_c = last_input_.shape()[0];
  size_t in_h = last_input_.shape()[1];
  size_t in_w = last_input_.shape()[2];

  size_t out_c = grad_output.shape()[0];
  size_t out_h = grad_output.shape()[1];
  size_t out_w = grad_output.shape()[2];

  // 初始化梯度
  Tensor input_grad(last_input_.shape());
  input_grad.zeros();
  weight_grad_.zeros();
  if (use_bias_) {
    bias_grad_.zeros();
  }

  // 计算偏置梯度（简单求和）
  if (use_bias_) {
    for (size_t oc = 0; oc < out_c; oc++) {
      float bias_grad_sum = 0.0f;
      for (size_t oh = 0; oh < out_h; oh++) {
        for (size_t ow = 0; ow < out_w; ow++) {
          size_t out_idx = oc * out_h * out_w + oh * out_w + ow;
          bias_grad_sum += grad_output[out_idx];
        }
      }
      bias_grad_[oc] += bias_grad_sum;
    }
  }

  // 计算权重梯度和输入梯度
  for (size_t oc = 0; oc < out_c; oc++) {
    for (size_t oh = 0; oh < out_h; oh++) {
      for (size_t ow = 0; ow < out_w; ow++) {
        size_t out_idx = oc * out_h * out_w + oh * out_w + ow;
        float grad_val = grad_output[out_idx];

        for (size_t ic = 0; ic < in_c; ic++) {
          for (int kh = 0; kh < kernel_size_; kh++) {
            for (int kw = 0; kw < kernel_size_; kw++) {
              int ih = oh * stride_ - padding_ + kh;
              int iw = ow * stride_ - padding_ + kw;

              if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
                size_t input_idx = ic * in_h * in_w + ih * in_w + iw;
                size_t weight_idx =
                    oc * in_channels_ * kernel_size_ * kernel_size_ +
                    ic * kernel_size_ * kernel_size_ + kh * kernel_size_ + kw;

                // 权重梯度：∂L/∂W = input * grad_output
                weight_grad_[weight_idx] += last_input_[input_idx] * grad_val;

                // 输入梯度：∂L/∂input = weight * grad_output
                input_grad[input_idx] += weights_[weight_idx] * grad_val;
              }
            }
          }
        }
      }
    }
  }

  return input_grad;
}

std::vector<Tensor *> ConvLayer::parameters() {
  std::vector<Tensor *> params;
  if (in_channels_ > 0) {
    params.push_back(&weights_);
    if (use_bias_) {
      params.push_back(&bias_);
    }
  }
  return params;
}

std::vector<Tensor *> ConvLayer::gradients() {
  std::vector<Tensor *> grads;
  if (in_channels_ > 0) {
    grads.push_back(&weight_grad_);
    if (use_bias_) {
      grads.push_back(&bias_grad_);
    }
  }
  return grads;
}

std::vector<int>
ConvLayer::output_shape(const std::vector<int> &input_shape) const {
  if (input_shape.size() != 4) {
    return input_shape;
  }
  int batch_size = input_shape[0];
  int height = input_shape[2];
  int width = input_shape[3];
  int out_height = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
  int out_width = (width + 2 * padding_ - kernel_size_) / stride_ + 1;
  return {batch_size, out_channels_, out_height, out_width};
}

// FullyConnectedLayer实现
FullyConnectedLayer::FullyConnectedLayer(int out_features, bool bias)
    : out_features_(out_features), use_bias_(bias), in_features_(0) {}

FullyConnectedLayer::FullyConnectedLayer(int in_features, int out_features,
                                         bool bias)
    : in_features_(in_features), out_features_(out_features), use_bias_(bias) {
  initialize_parameters();
}

void FullyConnectedLayer::initialize_parameters() {
  if (in_features_ > 0) {
    weights_ = Tensor({(size_t)in_features_, (size_t)out_features_});
    // Xavier初始化
    weights_.xavier_uniform(in_features_, out_features_);

    if (use_bias_) {
      bias_ = Tensor({(size_t)out_features_});
      bias_.zeros();
    }
    weight_grad_ = Tensor(weights_.shape());
    weight_grad_.zeros();
    if (use_bias_) {
      bias_grad_ = Tensor(bias_.shape());
      bias_grad_.zeros();
    }
  }
}

Tensor FullyConnectedLayer::forward(const Tensor &input) {
  last_input_ = input;

  // 如果还没有初始化参数，根据输入形状初始化
  if (in_features_ == 0) {
    in_features_ = input.size();
    initialize_parameters();
  }

  // 确保输入是1D的
  Tensor flat_input = input;
  if (input.ndim() > 1) {
    // 展平输入
    flat_input = Tensor({input.size()});
    for (size_t i = 0; i < input.size(); i++) {
      flat_input[i] = input[i];
    }
  }

  // 矩阵乘法 + 偏置
  Tensor output({(size_t)out_features_});
  output.zeros();

  for (int o = 0; o < out_features_; o++) {
    float sum = 0.0f;
    for (int i = 0; i < in_features_; i++) {
      sum += flat_input[i] * weights_[i * out_features_ + o];
    }
    if (use_bias_) {
      sum += bias_[o];
    }
    output[o] = sum;
  }

  return output;
}

Tensor FullyConnectedLayer::backward(const Tensor &grad_output) {
  if (last_input_.size() == 0 || in_features_ == 0) {
    return grad_output.clone();
  }

  // 确保输入是1D的
  Tensor flat_input = last_input_;
  if (last_input_.ndim() > 1) {
    flat_input = Tensor({last_input_.size()});
    for (size_t i = 0; i < last_input_.size(); i++) {
      flat_input[i] = last_input_[i];
    }
  }

  // 初始化梯度
  Tensor input_grad({(size_t)in_features_});
  input_grad.zeros();
  weight_grad_.zeros();
  if (use_bias_) {
    bias_grad_.zeros();
  }

  // 计算偏置梯度：∂L/∂b = grad_output
  if (use_bias_) {
    for (int o = 0; o < out_features_; o++) {
      bias_grad_[o] += grad_output[o];
    }
  }

  // 计算权重梯度和输入梯度
  for (int i = 0; i < in_features_; i++) {
    for (int o = 0; o < out_features_; o++) {
      size_t weight_idx = i * out_features_ + o;

      // 权重梯度：∂L/∂W = input * grad_output
      weight_grad_[weight_idx] += flat_input[i] * grad_output[o];

      // 输入梯度：∂L/∂input = weight * grad_output
      input_grad[i] += weights_[weight_idx] * grad_output[o];
    }
  }

  // 如果原始输入是多维的，reshape回原始形状
  if (last_input_.ndim() > 1) {
    Tensor reshaped_grad(last_input_.shape());
    for (size_t i = 0; i < last_input_.size(); i++) {
      reshaped_grad.data()[i] = input_grad[i];
    }
    return reshaped_grad;
  }

  return input_grad;
}

std::vector<Tensor *> FullyConnectedLayer::parameters() {
  std::vector<Tensor *> params;
  if (in_features_ > 0) {
    params.push_back(&weights_);
    if (use_bias_) {
      params.push_back(&bias_);
    }
  }
  return params;
}

std::vector<Tensor *> FullyConnectedLayer::gradients() {
  std::vector<Tensor *> grads;
  if (in_features_ > 0) {
    grads.push_back(&weight_grad_);
    if (use_bias_) {
      grads.push_back(&bias_grad_);
    }
  }
  return grads;
}

std::vector<int>
FullyConnectedLayer::output_shape(const std::vector<int> &input_shape) const {
  if (input_shape.empty()) {
    return {out_features_};
  }
  std::vector<int> output_shape = input_shape;
  output_shape.back() = out_features_;
  return output_shape;
}

// ReLULayer实现
Tensor ReLULayer::forward(const Tensor &input) {
  last_input_ = input;
  return input.relu();
}

Tensor ReLULayer::backward(const Tensor &grad_output) {
  // ReLU的导数：输入>0时为1，否则为0
  Tensor grad_input(grad_output.shape());
  for (size_t i = 0; i < grad_output.size(); i++) {
    grad_input[i] = (last_input_[i] > 0) ? grad_output[i] : 0.0f;
  }
  return grad_input;
}

std::vector<int>
ReLULayer::output_shape(const std::vector<int> &input_shape) const {
  return input_shape;
}

// SigmoidLayer实现
Tensor SigmoidLayer::forward(const Tensor &input) {
  last_output_ = input.sigmoid();
  return last_output_;
}

Tensor SigmoidLayer::backward(const Tensor &grad_output) {
  // Sigmoid的导数：σ(x) * (1 - σ(x))
  Tensor grad_input(grad_output.shape());
  for (size_t i = 0; i < grad_output.size(); i++) {
    float sigmoid_val = last_output_[i];
    grad_input[i] = grad_output[i] * sigmoid_val * (1.0f - sigmoid_val);
  }
  return grad_input;
}

std::vector<int>
SigmoidLayer::output_shape(const std::vector<int> &input_shape) const {
  return input_shape;
}

// TanhLayer实现
Tensor TanhLayer::forward(const Tensor &input) {
  last_output_ = input.tanh();
  return last_output_;
}

Tensor TanhLayer::backward(const Tensor &grad_output) {
  // Tanh的导数：1 - tanh²(x)
  Tensor grad_input(grad_output.shape());
  for (size_t i = 0; i < grad_output.size(); i++) {
    float tanh_val = last_output_[i];
    grad_input[i] = grad_output[i] * (1.0f - tanh_val * tanh_val);
  }
  return grad_input;
}

std::vector<int>
TanhLayer::output_shape(const std::vector<int> &input_shape) const {
  return input_shape;
}

// SoftmaxLayer实现
Tensor SoftmaxLayer::forward(const Tensor &input) {
  last_output_ = input.softmax();
  return last_output_;
}

Tensor SoftmaxLayer::backward(const Tensor &grad_output) {
  // 简化：对于交叉熵损失，softmax的梯度通常在损失函数中处理
  return grad_output;
}

std::vector<int>
SoftmaxLayer::output_shape(const std::vector<int> &input_shape) const {
  return input_shape;
}

// MaxPoolLayer实现
MaxPoolLayer::MaxPoolLayer(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride == -1 ? kernel_size : stride),
      padding_(padding) {}

Tensor MaxPoolLayer::forward(const Tensor &input) {
  last_input_ = input;

  if (input.ndim() != 3) {
    throw std::runtime_error("MaxPoolLayer期望3D输入 (C, H, W)");
  }

  size_t in_c = input.shape()[0];
  size_t in_h = input.shape()[1];
  size_t in_w = input.shape()[2];

  size_t out_h = (in_h + 2 * padding_ - kernel_size_) / stride_ + 1;
  size_t out_w = (in_w + 2 * padding_ - kernel_size_) / stride_ + 1;

  Tensor output({in_c, out_h, out_w});
  output.zeros();

  // 创建最大值索引张量来记录哪个位置被选中
  max_indices_ = Tensor({in_c, out_h, out_w});
  max_indices_.zeros();

  // MaxPool计算
  for (size_t c = 0; c < in_c; c++) {
    for (size_t oh = 0; oh < out_h; oh++) {
      for (size_t ow = 0; ow < out_w; ow++) {
        float max_val = -std::numeric_limits<float>::max();
        size_t max_idx = 0;

        for (int kh = 0; kh < kernel_size_; kh++) {
          for (int kw = 0; kw < kernel_size_; kw++) {
            int ih = oh * stride_ - padding_ + kh;
            int iw = ow * stride_ - padding_ + kw;

            if (ih >= 0 && ih < (int)in_h && iw >= 0 && iw < (int)in_w) {
              size_t input_idx = c * in_h * in_w + ih * in_w + iw;
              if (input[input_idx] > max_val) {
                max_val = input[input_idx];
                max_idx = input_idx;
              }
            }
          }
        }

        size_t output_idx = c * out_h * out_w + oh * out_w + ow;
        output[output_idx] = max_val;
        max_indices_[output_idx] = static_cast<float>(max_idx);
      }
    }
  }

  return output;
}

Tensor MaxPoolLayer::backward(const Tensor &grad_output) {
  if (last_input_.size() == 0) {
    return grad_output.clone();
  }

  Tensor input_grad(last_input_.shape());
  input_grad.zeros();

  size_t out_c = grad_output.shape()[0];
  size_t out_h = grad_output.shape()[1];
  size_t out_w = grad_output.shape()[2];

  // 将梯度传播回最大值位置
  for (size_t c = 0; c < out_c; c++) {
    for (size_t oh = 0; oh < out_h; oh++) {
      for (size_t ow = 0; ow < out_w; ow++) {
        size_t output_idx = c * out_h * out_w + oh * out_w + ow;
        size_t max_input_idx = static_cast<size_t>(max_indices_[output_idx]);

        // 只有最大值位置才接收梯度
        input_grad[max_input_idx] += grad_output[output_idx];
      }
    }
  }

  return input_grad;
}

std::vector<int>
MaxPoolLayer::output_shape(const std::vector<int> &input_shape) const {
  if (input_shape.size() != 4) {
    return input_shape;
  }
  int batch_size = input_shape[0];
  int channels = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];
  int out_height = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
  int out_width = (width + 2 * padding_ - kernel_size_) / stride_ + 1;
  return {batch_size, channels, out_height, out_width};
}

// AvgPoolLayer实现
AvgPoolLayer::AvgPoolLayer(int kernel_size, int stride, int padding)
    : kernel_size_(kernel_size), stride_(stride == -1 ? kernel_size : stride),
      padding_(padding) {}

Tensor AvgPoolLayer::forward(const Tensor &input) {
  last_input_ = input;
  return input.clone(); // 简化实现
}

Tensor AvgPoolLayer::backward(const Tensor &grad_output) {
  return grad_output.clone();
}

std::vector<int>
AvgPoolLayer::output_shape(const std::vector<int> &input_shape) const {
  if (input_shape.size() != 4) {
    return input_shape;
  }
  int batch_size = input_shape[0];
  int channels = input_shape[1];
  int height = input_shape[2];
  int width = input_shape[3];
  int out_height = (height + 2 * padding_ - kernel_size_) / stride_ + 1;
  int out_width = (width + 2 * padding_ - kernel_size_) / stride_ + 1;
  return {batch_size, channels, out_height, out_width};
}

// DropoutLayer实现
Tensor DropoutLayer::forward(const Tensor &input) {
  if (!is_training()) {
    return input;
  }
  return input.clone(); // 简化实现
}

Tensor DropoutLayer::backward(const Tensor &grad_output) {
  return grad_output.clone();
}

std::vector<int>
DropoutLayer::output_shape(const std::vector<int> &input_shape) const {
  return input_shape;
}

// BatchNormLayer实现
BatchNormLayer::BatchNormLayer(int num_features, float eps, float momentum)
    : num_features_(num_features), eps_(eps), momentum_(momentum) {
  initialize_parameters();
}

void BatchNormLayer::initialize_parameters() {
  gamma_ = Tensor({(size_t)num_features_});
  beta_ = Tensor({(size_t)num_features_});
  running_mean_ = Tensor({(size_t)num_features_});
  running_var_ = Tensor({(size_t)num_features_});
  gamma_grad_ = Tensor(gamma_.shape());
  beta_grad_ = Tensor(beta_.shape());
}

Tensor BatchNormLayer::forward(const Tensor &input) {
  last_input_ = input;
  return input.clone(); // 简化实现
}

Tensor BatchNormLayer::backward(const Tensor &grad_output) {
  return grad_output.clone();
}

std::vector<Tensor *> BatchNormLayer::parameters() { return {&gamma_, &beta_}; }

std::vector<Tensor *> BatchNormLayer::gradients() {
  return {&gamma_grad_, &beta_grad_};
}

std::vector<int>
BatchNormLayer::output_shape(const std::vector<int> &input_shape) const {
  return input_shape;
}

// FlattenLayer实现
Tensor FlattenLayer::forward(const Tensor &input) {
  last_input_shape_ =
      std::vector<int>(input.shape().begin(), input.shape().end());

  size_t total_elements = input.size();
  std::vector<size_t> new_shape;

  if (last_input_shape_.empty()) {
    new_shape = {total_elements};
  } else {
    new_shape = {(size_t)last_input_shape_[0],
                 total_elements / (size_t)last_input_shape_[0]};
  }

  Tensor result = input.clone();
  result.reshape(new_shape);
  return result;
}

Tensor FlattenLayer::backward(const Tensor &grad_output) {
  std::vector<size_t> orig_shape(last_input_shape_.begin(),
                                 last_input_shape_.end());
  Tensor result = grad_output.clone();
  result.reshape(orig_shape);
  return result;
}

std::vector<int>
FlattenLayer::output_shape(const std::vector<int> &input_shape) const {
  if (input_shape.empty()) {
    return {};
  }

  int total_elements = 1;
  for (int dim : input_shape) {
    total_elements *= dim;
  }

  return {input_shape[0], total_elements / input_shape[0]};
}

} // namespace CNN