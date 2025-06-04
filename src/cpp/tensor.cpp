#include "cnn/tensor.h"
#include "cnn_core/math_core.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace CNN {

// 构造函数和析构函数实现
Tensor::Tensor() {
  // 创建一个空张量
  std::memset(&core_tensor_, 0, sizeof(core_tensor_));
  core_tensor_.owns_data = 1;
}

Tensor::Tensor(const std::vector<size_t> &dims) {
  // 创建具有指定形状的张量
  std::memset(&core_tensor_, 0, sizeof(core_tensor_));
  if (dims.size() > 0) {
    cnn_core_tensor_create(&core_tensor_, dims.data(), dims.size());
  }
  update_shape_cache();
}

Tensor::Tensor(std::initializer_list<size_t> dims)
    : Tensor(std::vector<size_t>(dims)) {}

Tensor::Tensor(const Tensor &other) {
  // 拷贝构造函数
  std::memset(&core_tensor_, 0, sizeof(core_tensor_));
  if (other.size() > 0) {
    cnn_core_tensor_create(&core_tensor_, other.shape().data(),
                           other.shape().size());
    if (core_tensor_.size > 0 && other.core_tensor_.data) {
      std::memcpy(core_tensor_.data, other.core_tensor_.data,
                  core_tensor_.size * sizeof(float));
    }
  }
  update_shape_cache();
}

Tensor::Tensor(Tensor &&other) noexcept {
  // 移动构造函数
  std::memset(&core_tensor_, 0, sizeof(core_tensor_));

  // 转移所有权
  core_tensor_ = other.core_tensor_;
  shape_ = std::move(other.shape_);

  // 清空源对象
  std::memset(&other.core_tensor_, 0, sizeof(cnn_core_tensor_t));
  other.shape_.clear();
}

Tensor::Tensor(const cnn_core_tensor_t &core_tensor, bool take_ownership) {
  // 从C核心张量构造
  if (!take_ownership) {
    // 创建副本
    std::memset(&core_tensor_, 0, sizeof(core_tensor_));
    cnn_core_tensor_create(&core_tensor_, core_tensor.dims, core_tensor.ndim);
    if (core_tensor_.data && core_tensor.data) {
      std::memcpy(core_tensor_.data, core_tensor.data,
                  core_tensor.size * sizeof(float));
    }
  } else {
    // 直接使用现有张量
    core_tensor_ = core_tensor;
  }
  update_shape_cache();
}

Tensor::~Tensor() {
  // 清理资源
  if (core_tensor_.data && core_tensor_.owns_data) {
    cnn_core_tensor_destroy(&core_tensor_);
  }
}

// 赋值操作符
Tensor &Tensor::operator=(const Tensor &other) {
  if (this != &other) {
    // 清理当前资源
    if (core_tensor_.data && core_tensor_.owns_data) {
      cnn_core_tensor_destroy(&core_tensor_);
    }

    // 拷贝
    std::memset(&core_tensor_, 0, sizeof(core_tensor_));
    if (other.size() > 0) {
      cnn_core_tensor_create(&core_tensor_, other.shape().data(),
                             other.shape().size());
      if (core_tensor_.size > 0 && other.core_tensor_.data) {
        std::memcpy(core_tensor_.data, other.core_tensor_.data,
                    core_tensor_.size * sizeof(float));
      }
    }
    update_shape_cache();
  }
  return *this;
}

Tensor &Tensor::operator=(Tensor &&other) noexcept {
  if (this != &other) {
    // 清理当前资源
    if (core_tensor_.data && core_tensor_.owns_data) {
      cnn_core_tensor_destroy(&core_tensor_);
    }

    // 转移所有权
    core_tensor_ = other.core_tensor_;
    shape_ = std::move(other.shape_);

    // 清空源对象
    std::memset(&other.core_tensor_, 0, sizeof(cnn_core_tensor_t));
    other.shape_.clear();
  }
  return *this;
}

// 操作实现
Tensor &Tensor::reshape(const std::vector<size_t> &new_dims) {
  if (!core_tensor_.data)
    return *this;

  // 计算新大小
  size_t new_size = 1;
  for (size_t dim : new_dims) {
    new_size *= dim;
  }

  // 检查大小是否匹配
  if (new_size != core_tensor_.size) {
    throw std::runtime_error("Reshape错误：新形状元素总数与原始张量不匹配");
  }

  // 应用重塑
  cnn_core_tensor_reshape(&core_tensor_, new_dims.data(), new_dims.size());
  update_shape_cache();

  return *this;
}

Tensor &Tensor::reshape(std::initializer_list<size_t> new_dims) {
  return reshape(std::vector<size_t>(new_dims));
}

Tensor &Tensor::fill(float value) {
  // 填充值
  if (core_tensor_.data) {
    cnn_core_tensor_fill(&core_tensor_, value);
  }
  return *this;
}

Tensor &Tensor::zeros() { return fill(0.0f); }

Tensor &Tensor::ones() { return fill(1.0f); }

Tensor &Tensor::rand(float min, float max, unsigned int seed) {
  if (!core_tensor_.data)
    return *this;

  // 使用随机数引擎
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(min, max);

  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] = dist(rng);
  }

  return *this;
}

// 算术运算
Tensor Tensor::operator+(const Tensor &other) const {
  if (!core_tensor_.data || !other.core_tensor_.data) {
    throw std::runtime_error("加法错误：张量未初始化");
  }

  if (size() != other.size()) {
    throw std::runtime_error("加法错误：张量形状不匹配");
  }

  // 创建结果张量
  Tensor result(shape());

  // 执行加法
  cnn_core_tensor_add(&result.core_tensor_, &core_tensor_, &other.core_tensor_);

  return result;
}

Tensor &Tensor::operator+=(const Tensor &other) {
  if (!core_tensor_.data || !other.core_tensor_.data) {
    throw std::runtime_error("加法错误：张量未初始化");
  }

  if (size() != other.size()) {
    throw std::runtime_error("加法错误：张量形状不匹配");
  }

  // 执行加法
  cnn_core_tensor_add(&core_tensor_, &core_tensor_, &other.core_tensor_);

  return *this;
}

Tensor Tensor::operator*(const Tensor &other) const {
  if (!core_tensor_.data || !other.core_tensor_.data) {
    throw std::runtime_error("乘法错误：张量未初始化");
  }

  if (size() != other.size()) {
    throw std::runtime_error("乘法错误：张量形状不匹配");
  }

  // 创建结果张量
  Tensor result(shape());

  // 执行元素乘法
  cnn_core_tensor_mul(&result.core_tensor_, &core_tensor_, &other.core_tensor_);

  return result;
}

Tensor Tensor::operator*(float scalar) const {
  if (!core_tensor_.data) {
    throw std::runtime_error("标量乘法错误：张量未初始化");
  }

  // 创建结果张量
  Tensor result(shape());

  // 执行标量乘法
  cnn_core_tensor_scale(&result.core_tensor_, &core_tensor_, scalar);

  return result;
}

float Tensor::operator[](size_t index) const {
  if (!core_tensor_.data || !core_tensor_.data || index >= core_tensor_.size) {
    throw std::out_of_range("索引越界");
  }

  return core_tensor_.data[index];
}

float &Tensor::operator[](size_t index) {
  if (!core_tensor_.data || !core_tensor_.data || index >= core_tensor_.size) {
    throw std::out_of_range("索引越界");
  }

  return core_tensor_.data[index];
}

// 获取张量信息
size_t Tensor::ndim() const { return core_tensor_.ndim; }
size_t Tensor::size() const { return core_tensor_.size; }

void Tensor::update_shape_cache() {
  shape_.clear();
  if (core_tensor_.ndim > 0) {
    shape_.resize(core_tensor_.ndim);
    for (size_t i = 0; i < core_tensor_.ndim; ++i) {
      shape_[i] = core_tensor_.dims[i];
    }
  }
}

const std::vector<size_t> &Tensor::shape() const { return shape_; }

// 矩阵运算
Tensor Tensor::matmul(const Tensor &other) const {
  if (!core_tensor_.data || !other.core_tensor_.data) {
    throw std::runtime_error("矩阵乘法错误：张量未初始化");
  }

  if (ndim() != 2 || other.ndim() != 2 || shape()[1] != other.shape()[0]) {
    throw std::runtime_error("矩阵乘法错误：张量形状不兼容");
  }

  // 创建结果张量
  Tensor result({shape()[0], other.shape()[1]});

  // 执行矩阵乘法
  cnn_core_tensor_matmul(&result.core_tensor_, &core_tensor_,
                         &other.core_tensor_);

  return result;
}

// 激活函数
Tensor Tensor::relu() const {
  // 创建副本
  Tensor result(*this);

  // 应用ReLU函数
  if (result.core_tensor_.data) {
    for (size_t i = 0; i < result.core_tensor_.size; ++i) {
      result.core_tensor_.data[i] = std::max(0.0f, result.core_tensor_.data[i]);
    }
  }

  return result;
}

Tensor Tensor::sigmoid() const {
  // 创建副本
  Tensor result(*this);

  // 应用Sigmoid函数
  if (result.core_tensor_.data) {
    for (size_t i = 0; i < result.core_tensor_.size; ++i) {
      result.core_tensor_.data[i] =
          1.0f / (1.0f + std::exp(-result.core_tensor_.data[i]));
    }
  }

  return result;
}

Tensor Tensor::tanh() const {
  // 创建副本
  Tensor result(*this);

  // 应用Tanh函数
  if (result.core_tensor_.data) {
    for (size_t i = 0; i < result.core_tensor_.size; ++i) {
      result.core_tensor_.data[i] = std::tanh(result.core_tensor_.data[i]);
    }
  }

  return result;
}

Tensor Tensor::softmax() const {
  // 创建副本
  Tensor result(*this);

  // 应用Softmax函数
  if (result.core_tensor_.data) {
    // 简化实现，假设是1D或2D张量
    size_t batch_size = 1;
    if (ndim() > 1) {
      batch_size = shape()[0];
    }

    size_t features = size() / batch_size;

    for (size_t b = 0; b < batch_size; ++b) {
      // 找到最大值
      float max_val = -std::numeric_limits<float>::max();
      for (size_t f = 0; f < features; ++f) {
        size_t idx = b * features + f;
        max_val = std::max(max_val, core_tensor_.data[idx]);
      }

      // 计算指数和
      float sum = 0.0f;
      for (size_t f = 0; f < features; ++f) {
        size_t idx = b * features + f;
        result.core_tensor_.data[idx] =
            std::exp(core_tensor_.data[idx] - max_val);
        sum += result.core_tensor_.data[idx];
      }

      // 归一化
      for (size_t f = 0; f < features; ++f) {
        size_t idx = b * features + f;
        result.core_tensor_.data[idx] /= sum;
      }
    }
  }

  return result;
}

// 其他方法
Tensor Tensor::clone() const { return Tensor(*this); }

// 数据访问方法
float *Tensor::data() { return core_tensor_.data; }

const float *Tensor::data() const { return core_tensor_.data; }

// 获取C核心张量
const cnn_core_tensor_t &Tensor::get_core_tensor() const {
  return core_tensor_;
}

void Tensor::print() const {
  if (!core_tensor_.data) {
    std::cout << "Tensor [未初始化]" << std::endl;
    return;
  }

  std::cout << "Tensor [";
  for (size_t i = 0; i < ndim(); ++i) {
    std::cout << shape()[i];
    if (i < ndim() - 1) {
      std::cout << ", ";
    }
  }
  std::cout << "], 大小: " << core_tensor_.size << std::endl;

  // 打印数据，简化实现仅适用于小张量
  if (size() <= 100) {
    for (size_t i = 0; i < std::min(size(), (size_t)10); ++i) {
      std::cout << core_tensor_.data[i] << " ";
    }
    if (size() > 10) {
      std::cout << "...";
    }
    std::cout << std::endl;
  }
}

// 全局运算符
Tensor operator*(float scalar, const Tensor &tensor) { return tensor * scalar; }

// 添加缺少的方法实现
float Tensor::get(const std::vector<size_t> &indices) const {
  if (!core_tensor_.data) {
    throw std::runtime_error("张量未初始化");
  }

  if (indices.size() != core_tensor_.ndim) {
    throw std::runtime_error("索引维度不匹配");
  }

  // 计算一维索引
  size_t flat_index = 0;
  size_t stride = 1;
  for (int i = core_tensor_.ndim - 1; i >= 0; --i) {
    if (indices[i] >= core_tensor_.dims[i]) {
      throw std::out_of_range("索引越界");
    }
    flat_index += indices[i] * stride;
    stride *= core_tensor_.dims[i];
  }

  return core_tensor_.data[flat_index];
}

void Tensor::set(const std::vector<size_t> &indices, float value) {
  if (!core_tensor_.data) {
    throw std::runtime_error("张量未初始化");
  }

  if (indices.size() != core_tensor_.ndim) {
    throw std::runtime_error("索引维度不匹配");
  }

  // 计算一维索引
  size_t flat_index = 0;
  size_t stride = 1;
  for (int i = core_tensor_.ndim - 1; i >= 0; --i) {
    if (indices[i] >= core_tensor_.dims[i]) {
      throw std::out_of_range("索引越界");
    }
    flat_index += indices[i] * stride;
    stride *= core_tensor_.dims[i];
  }

  core_tensor_.data[flat_index] = value;
}

Tensor Tensor::operator-(const Tensor &other) const {
  if (!core_tensor_.data || !other.core_tensor_.data) {
    throw std::runtime_error("减法错误：张量未初始化");
  }

  if (size() != other.size()) {
    throw std::runtime_error("减法错误：张量形状不匹配");
  }

  // 创建结果张量
  Tensor result(shape());

  // 执行减法
  for (size_t i = 0; i < result.core_tensor_.size; ++i) {
    result.core_tensor_.data[i] =
        core_tensor_.data[i] - other.core_tensor_.data[i];
  }

  return result;
}

Tensor &Tensor::operator-=(const Tensor &other) {
  if (!core_tensor_.data || !other.core_tensor_.data) {
    throw std::runtime_error("减法错误：张量未初始化");
  }

  if (size() != other.size()) {
    throw std::runtime_error("减法错误：张量形状不匹配");
  }

  // 执行原地减法
  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] -= other.core_tensor_.data[i];
  }

  return *this;
}

Tensor &Tensor::operator*=(float scalar) {
  if (!core_tensor_.data) {
    throw std::runtime_error("标量乘法错误：张量未初始化");
  }

  // 执行原地标量乘法
  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] *= scalar;
  }

  return *this;
}

Tensor &Tensor::operator*=(const Tensor &other) {
  if (!core_tensor_.data || !other.core_tensor_.data) {
    throw std::runtime_error("乘法错误：张量未初始化");
  }

  if (size() != other.size()) {
    throw std::runtime_error("乘法错误：张量形状不匹配");
  }

  // 执行原地元素乘法
  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] *= other.core_tensor_.data[i];
  }

  return *this;
}

Tensor Tensor::transpose() const {
  if (!core_tensor_.data) {
    throw std::runtime_error("转置错误：张量未初始化");
  }

  if (core_tensor_.ndim != 2) {
    throw std::runtime_error("转置错误：只支持2D张量");
  }

  size_t rows = core_tensor_.dims[0];
  size_t cols = core_tensor_.dims[1];

  // 创建转置结果张量
  Tensor result({cols, rows});

  // 执行转置
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      result.core_tensor_.data[j * rows + i] = core_tensor_.data[i * cols + j];
    }
  }

  return result;
}

std::string Tensor::to_string() const {
  if (!core_tensor_.data) {
    return "Tensor [未初始化]";
  }

  std::ostringstream oss;
  oss << "Tensor(shape=[";
  for (size_t i = 0; i < core_tensor_.ndim; ++i) {
    if (i > 0)
      oss << ", ";
    oss << core_tensor_.dims[i];
  }
  oss << "], data=[";

  size_t max_elements = std::min(size_t(10), core_tensor_.size);
  for (size_t i = 0; i < max_elements; ++i) {
    if (i > 0)
      oss << ", ";
    oss << core_tensor_.data[i];
  }
  if (core_tensor_.size > max_elements) {
    oss << ", ...";
  }
  oss << "])";

  return oss.str();
}

Tensor Tensor::apply(const std::function<float(float)> &func) const {
  if (!core_tensor_.data) {
    throw std::runtime_error("应用函数错误：张量未初始化");
  }

  Tensor result(shape());

  for (size_t i = 0; i < core_tensor_.size; ++i) {
    result.core_tensor_.data[i] = func(core_tensor_.data[i]);
  }

  return result;
}

Tensor &Tensor::apply_inplace(const std::function<float(float)> &func) {
  if (!core_tensor_.data) {
    throw std::runtime_error("原地应用函数错误：张量未初始化");
  }

  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] = func(core_tensor_.data[i]);
  }

  return *this;
}

Tensor &Tensor::relu_inplace() {
  if (!core_tensor_.data) {
    throw std::runtime_error("ReLU错误：张量未初始化");
  }

  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] = std::max(0.0f, core_tensor_.data[i]);
  }

  return *this;
}

Tensor &Tensor::sigmoid_inplace() {
  if (!core_tensor_.data) {
    throw std::runtime_error("Sigmoid错误：张量未初始化");
  }

  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] = 1.0f / (1.0f + std::exp(-core_tensor_.data[i]));
  }

  return *this;
}

Tensor &Tensor::tanh_inplace() {
  if (!core_tensor_.data) {
    throw std::runtime_error("Tanh错误：张量未初始化");
  }

  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] = std::tanh(core_tensor_.data[i]);
  }

  return *this;
}

Tensor &Tensor::softmax_inplace() {
  if (!core_tensor_.data) {
    throw std::runtime_error("Softmax错误：张量未初始化");
  }

  // 简化实现，假设是1D或2D张量
  size_t batch_size = 1;
  if (ndim() > 1) {
    batch_size = shape()[0];
  }

  size_t features = size() / batch_size;

  for (size_t b = 0; b < batch_size; ++b) {
    // 找到最大值
    float max_val = -std::numeric_limits<float>::max();
    for (size_t f = 0; f < features; ++f) {
      size_t idx = b * features + f;
      max_val = std::max(max_val, core_tensor_.data[idx]);
    }

    // 计算指数和
    float sum = 0.0f;
    for (size_t f = 0; f < features; ++f) {
      size_t idx = b * features + f;
      core_tensor_.data[idx] = std::exp(core_tensor_.data[idx] - max_val);
      sum += core_tensor_.data[idx];
    }

    // 归一化
    for (size_t f = 0; f < features; ++f) {
      size_t idx = b * features + f;
      core_tensor_.data[idx] /= sum;
    }
  }

  return *this;
}

Tensor &Tensor::randn(float mean, float stddev, unsigned int seed) {
  if (!core_tensor_.data) {
    return *this;
  }

  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(mean, stddev);

  for (size_t i = 0; i < core_tensor_.size; ++i) {
    core_tensor_.data[i] = dist(rng);
  }

  return *this;
}

Tensor &Tensor::xavier_uniform(size_t fan_in, size_t fan_out) {
  if (!core_tensor_.data) {
    return *this;
  }

  if (fan_in == 0 || fan_out == 0) {
    // 自动推断输入输出单元数
    if (ndim() >= 2) {
      fan_in = shape()[0];
      fan_out = shape()[1];
    } else {
      fan_in = fan_out = size();
    }
  }

  float limit = std::sqrt(6.0f / (fan_in + fan_out));
  return rand(-limit, limit);
}

} // namespace CNN