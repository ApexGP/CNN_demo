#include "cnn/layers.h"
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
    if (use_bias_) {
      bias_ = Tensor({(size_t)out_channels_});
    }
    weight_grad_ = Tensor(weights_.shape());
    if (use_bias_) {
      bias_grad_ = Tensor(bias_.shape());
    }
  }
}

Tensor ConvLayer::forward(const Tensor &input) {
  last_input_ = input;
  return input.clone(); // 简化实现
}

Tensor ConvLayer::backward(const Tensor &grad_output) {
  return grad_output.clone();
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
    if (use_bias_) {
      bias_ = Tensor({(size_t)out_features_});
    }
    weight_grad_ = Tensor(weights_.shape());
    if (use_bias_) {
      bias_grad_ = Tensor(bias_.shape());
    }
  }
}

Tensor FullyConnectedLayer::forward(const Tensor &input) {
  last_input_ = input;
  if (in_features_ == 0) {
    in_features_ = input.size() / input.shape()[0];
    initialize_parameters();
  }
  return input.clone(); // 简化实现
}

Tensor FullyConnectedLayer::backward(const Tensor &grad_output) {
  return grad_output.clone();
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
  return grad_output.clone(); // 简化实现
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
  return grad_output.clone(); // 简化实现
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
  return grad_output.clone(); // 简化实现
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

Tensor SoftmaxLayer::backward(const Tensor &grad_output) { return grad_output; }

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
  return input.clone(); // 简化实现
}

Tensor MaxPoolLayer::backward(const Tensor &grad_output) {
  return grad_output.clone();
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