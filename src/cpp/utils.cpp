/**
 * @file utils.cpp
 * @brief CNN框架工具函数实现
 */

#include "cnn/utils.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace CNN {

// 数学工具函数实现
namespace Math {

float relu(float x) { return std::max(0.0f, x); }

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

float tanh_activation(float x) { return std::tanh(x); }

void xavier_uniform_init(float *data, int size, int fan_in, int fan_out) {
  if (!data || size <= 0)
    return;

  float limit = std::sqrt(6.0f / (fan_in + fan_out));
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-limit, limit);

  for (int i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
}

void kaiming_uniform_init(float *data, int size, int fan_in) {
  if (!data || size <= 0)
    return;

  float limit = std::sqrt(6.0f / fan_in);
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(-limit, limit);

  for (int i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
}

void normal_init(float *data, int size, float mean, float std) {
  if (!data || size <= 0)
    return;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dis(mean, std);

  for (int i = 0; i < size; ++i) {
    data[i] = dis(gen);
  }
}

float mean(const float *data, int size) {
  if (!data || size <= 0)
    return 0.0f;

  float sum = 0.0f;
  for (int i = 0; i < size; ++i) {
    sum += data[i];
  }
  return sum / size;
}

float variance(const float *data, int size) {
  if (!data || size <= 0)
    return 0.0f;

  float m = mean(data, size);
  float sum_sq_diff = 0.0f;

  for (int i = 0; i < size; ++i) {
    float diff = data[i] - m;
    sum_sq_diff += diff * diff;
  }

  return sum_sq_diff / size;
}

} // namespace Math

// 可视化工具实现
namespace Visualization {

void plot_training_history(const std::vector<float> &train_losses,
                           const std::vector<float> &val_losses,
                           const std::string &save_path) {
  std::cout << "=== 训练历史 ===" << std::endl;
  std::cout << "训练损失: ";
  for (size_t i = 0; i < std::min(size_t(10), train_losses.size()); ++i) {
    std::cout << train_losses[i] << " ";
  }
  if (train_losses.size() > 10)
    std::cout << "...";
  std::cout << std::endl;

  if (!save_path.empty()) {
    std::cout << "注意：保存到 " << save_path << " 功能需要matplotlib支持"
              << std::endl;
  }
}

void visualize_feature_maps(const Tensor &feature_maps,
                            const std::string &save_path) {
  std::cout << "=== 特征图可视化 ===" << std::endl;
  std::cout << "特征图形状: [";
  for (size_t i = 0; i < feature_maps.shape().size(); ++i) {
    if (i > 0)
      std::cout << ", ";
    std::cout << feature_maps.shape()[i];
  }
  std::cout << "]" << std::endl;
}

} // namespace Visualization

// 性能分析工具实现
namespace Profiler {

Timer::Timer() : running_(false) {}

void Timer::start() {
  start_time_ = std::chrono::high_resolution_clock::now();
  running_ = true;
}

void Timer::stop() {
  if (running_) {
    end_time_ = std::chrono::high_resolution_clock::now();
    running_ = false;
  }
}

double Timer::elapsed_milliseconds() const {
  auto end = running_ ? std::chrono::high_resolution_clock::now() : end_time_;
  auto duration =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start_time_);
  return duration.count() / 1000.0;
}

void Timer::reset() { running_ = false; }

} // namespace Profiler

// 系统信息实现
namespace SystemInfo {

HardwareInfo get_hardware_info() {
  HardwareInfo info;

  info.cpu_cores = std::thread::hardware_concurrency();
  info.cpu_model = "CPU";
  info.total_memory = 8ULL * 1024 * 1024 * 1024;

#ifdef _OPENMP
  info.has_openmp = true;
#else
  info.has_openmp = false;
#endif

#ifdef USE_CUDA
  info.has_cuda = true;
  info.gpu_model = "CUDA GPU";
#else
  info.has_cuda = false;
  info.gpu_model = "No GPU";
#endif

  return info;
}

void print_hardware_info() {
  HardwareInfo info = get_hardware_info();

  std::cout << "=== 硬件信息 ===" << std::endl;
  std::cout << "CPU型号: " << info.cpu_model << std::endl;
  std::cout << "CPU核心数: " << info.cpu_cores << std::endl;
  std::cout << "总内存: " << (info.total_memory / (1024 * 1024)) << " MB"
            << std::endl;
  std::cout << "OpenMP支持: " << (info.has_openmp ? "是" : "否") << std::endl;
  std::cout << "CUDA支持: " << (info.has_cuda ? "是" : "否") << std::endl;
  std::cout << "GPU型号: " << info.gpu_model << std::endl;
}

} // namespace SystemInfo

} // namespace CNN