#pragma once

#include "tensor.h"
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace CNN {

// 设备类型枚举
enum class Device { CPU, GPU };

// 数学工具
namespace Math {
// 激活函数
float relu(float x);
float sigmoid(float x);
float tanh_activation(float x);

// 初始化函数
void xavier_uniform_init(float *data, int size, int fan_in, int fan_out);
void kaiming_uniform_init(float *data, int size, int fan_in);
void normal_init(float *data, int size, float mean = 0.0f, float std = 1.0f);

// 统计函数
float mean(const float *data, int size);
float variance(const float *data, int size);
} // namespace Math

// 可视化工具
namespace Visualization {
void plot_training_history(const std::vector<float> &train_losses,
                           const std::vector<float> &val_losses,
                           const std::string &save_path = "");

void visualize_feature_maps(const Tensor &feature_maps,
                            const std::string &save_path = "");
} // namespace Visualization

// 性能分析工具
namespace Profiler {
class Timer {
public:
  Timer();
  void start();
  void stop();
  double elapsed_milliseconds() const;
  void reset();

private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;
  bool running_;
};
} // namespace Profiler

// 系统信息
namespace SystemInfo {
struct HardwareInfo {
  std::string cpu_model;
  int cpu_cores;
  size_t total_memory;
  bool has_cuda;
  std::string gpu_model;
  bool has_openmp;
};

HardwareInfo get_hardware_info();
void print_hardware_info();
} // namespace SystemInfo

} // namespace CNN