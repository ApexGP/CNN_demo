/**
 * @file cnn_module.cpp
 * @brief CNN框架的Python绑定
 */

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cnn/layers.h"
#include "cnn/network.h"
#include "cnn/optimizer.h"
#include "cnn/tensor.h"
#include "cnn/utils.h"

namespace py = pybind11;

// 张量类的绑定声明
void bind_tensor(py::module_ &m);
// 网络类的绑定声明
void bind_network(py::module_ &m);
// 层类的绑定声明
void bind_layers(py::module_ &m);

PYBIND11_MODULE(cnn_framework, m) {
  m.doc() = "CNN混合架构框架的Python绑定";

  m.attr("__version__") = "0.1.0";

  // 创建子模块
  py::module_ tensor_module = m.def_submodule("tensor", "张量操作模块");
  py::module_ network_module = m.def_submodule("network", "网络模块");
  py::module_ layers_module = m.def_submodule("layers", "网络层模块");
  py::module_ utils_module = m.def_submodule("utils", "工具函数模块");

  // 绑定张量类
  bind_tensor(m);

  // 绑定网络类
  bind_network(m);

  // 绑定层类
  bind_layers(m);

  // 绑定优化器类型枚举
  py::enum_<CNN::OptimizerType>(m, "OptimizerType")
      .value("SGD", CNN::OPTIMIZER_SGD)
      .value("MOMENTUM", CNN::OPTIMIZER_MOMENTUM)
      .value("ADAGRAD", CNN::OPTIMIZER_ADAGRAD)
      .value("RMSPROP", CNN::OPTIMIZER_RMSPROP)
      .value("ADAM", CNN::OPTIMIZER_ADAM)
      .export_values();

  // 绑定损失函数类型枚举
  py::enum_<CNN::LossType>(m, "LossType")
      .value("MSE", CNN::LOSS_MSE)
      .value("CROSS_ENTROPY", CNN::LOSS_CROSS_ENTROPY)
      .value("BINARY_CROSS_ENTROPY", CNN::LOSS_BINARY_CROSS_ENTROPY)
      .export_values();

  // 绑定设备枚举
  py::enum_<CNN::Device>(utils_module, "Device")
      .value("CPU", CNN::Device::CPU)
      .value("GPU", CNN::Device::GPU)
      .export_values();

  // 绑定数学工具函数
  py::module_ math_module = utils_module.def_submodule("math", "数学工具");
  math_module.def("relu", &CNN::Math::relu, "ReLU激活函数");
  math_module.def("sigmoid", &CNN::Math::sigmoid, "Sigmoid激活函数");
  math_module.def("tanh", &CNN::Math::tanh_activation, "Tanh激活函数");
  math_module.def("mean", &CNN::Math::mean, "计算均值");
  math_module.def("variance", &CNN::Math::variance, "计算方差");

  // 绑定系统信息
  py::module_ sysinfo_module =
      utils_module.def_submodule("sysinfo", "系统信息");
  py::class_<CNN::SystemInfo::HardwareInfo>(sysinfo_module, "HardwareInfo")
      .def_readonly("cpu_model", &CNN::SystemInfo::HardwareInfo::cpu_model)
      .def_readonly("cpu_cores", &CNN::SystemInfo::HardwareInfo::cpu_cores)
      .def_readonly("total_memory",
                    &CNN::SystemInfo::HardwareInfo::total_memory)
      .def_readonly("has_cuda", &CNN::SystemInfo::HardwareInfo::has_cuda)
      .def_readonly("gpu_model", &CNN::SystemInfo::HardwareInfo::gpu_model)
      .def_readonly("has_openmp", &CNN::SystemInfo::HardwareInfo::has_openmp);

  sysinfo_module.def("get_hardware_info", &CNN::SystemInfo::get_hardware_info,
                     "获取硬件信息");
  sysinfo_module.def("print_hardware_info",
                     &CNN::SystemInfo::print_hardware_info, "打印硬件信息");

  // 绑定计时器
  py::class_<CNN::Profiler::Timer>(utils_module, "Timer")
      .def(py::init<>())
      .def("start", &CNN::Profiler::Timer::start, "开始计时")
      .def("stop", &CNN::Profiler::Timer::stop, "停止计时")
      .def("elapsed_milliseconds", &CNN::Profiler::Timer::elapsed_milliseconds,
           "获取已用时间(毫秒)")
      .def("reset", &CNN::Profiler::Timer::reset, "重置计时器");

  // 简单的功能检查函数
  utils_module.def(
      "has_cuda",
      []() {
#ifdef USE_CUDA
        return true;
#else
        return false;
#endif
      },
      "检查是否支持CUDA");

  utils_module.def(
      "has_openblas",
      []() {
#ifdef USE_OPENBLAS
        return true;
#else
        return false;
#endif
      },
      "检查是否支持OpenBLAS");
}