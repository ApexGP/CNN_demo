/**
 * @file network_binding.cpp
 * @brief 网络类的Python绑定实现
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cnn/network.h"

namespace py = pybind11;

void bind_network(py::module_ &m) {
  // 绑定网络类
  py::class_<CNN::Network>(m, "Network")
      .def(py::init<>(), "创建空的神经网络")
      .def("add_conv_layer", &CNN::Network::add_conv_layer, "添加卷积层",
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0,
           py::arg("bias") = true)
      .def("add_fc_layer", &CNN::Network::add_fc_layer, "添加全连接层",
           py::arg("out_features"), py::arg("bias") = true)
      .def("add_relu_layer", &CNN::Network::add_relu_layer, "添加ReLU激活层")
      .def("add_sigmoid_layer", &CNN::Network::add_sigmoid_layer,
           "添加Sigmoid激活层")
      .def("add_tanh_layer", &CNN::Network::add_tanh_layer, "添加Tanh激活层")
      .def("add_softmax_layer", &CNN::Network::add_softmax_layer,
           "添加Softmax激活层", py::arg("dim") = -1)
      .def("add_maxpool_layer", &CNN::Network::add_maxpool_layer,
           "添加最大池化层", py::arg("kernel_size"), py::arg("stride") = -1,
           py::arg("padding") = 0)
      .def("add_dropout_layer", &CNN::Network::add_dropout_layer,
           "添加Dropout层", py::arg("p") = 0.5f)
      .def("add_flatten_layer", &CNN::Network::add_flatten_layer, "添加展平层")
      .def("forward", &CNN::Network::forward, "前向传播")
      .def("predict", &CNN::Network::predict, "预测")
      .def("predict_batch", &CNN::Network::predict_batch, "批量预测")
      .def("train",
           py::overload_cast<const std::vector<CNN::Tensor> &,
                             const std::vector<CNN::Tensor> &, int, int, float>(
               &CNN::Network::train),
           "训练网络", py::arg("train_data"), py::arg("train_labels"),
           py::arg("epochs") = 100, py::arg("batch_size") = 32,
           py::arg("learning_rate") = 0.001f)
      .def("train_with_validation", &CNN::Network::train_with_validation,
           "带验证的训练", py::arg("train_data"), py::arg("train_labels"),
           py::arg("val_data"), py::arg("val_labels"), py::arg("epochs") = 100,
           py::arg("batch_size") = 32, py::arg("learning_rate") = 0.001f)
      .def("evaluate", &CNN::Network::evaluate, "评估网络")
      .def("calculate_accuracy", &CNN::Network::calculate_accuracy,
           "计算准确率")
      .def("train_mode", &CNN::Network::train_mode, "设置为训练模式")
      .def("eval_mode", &CNN::Network::eval_mode, "设置为评估模式")
      .def("to_cpu", &CNN::Network::to_cpu, "移动到CPU")
      .def("to_gpu", &CNN::Network::to_gpu, "移动到GPU")
      .def("get_device", &CNN::Network::get_device, "获取当前设备")
      .def("save_model", &CNN::Network::save_model, "保存模型")
      .def("load_model", &CNN::Network::load_model, "加载模型")
      .def("save_weights", &CNN::Network::save_weights, "保存权重")
      .def("load_weights", &CNN::Network::load_weights, "加载权重")
      .def("print_summary", &CNN::Network::print_summary, "打印网络摘要")
      .def("get_num_parameters", &CNN::Network::get_num_parameters,
           "获取参数数量")
      .def("set_weight_decay", &CNN::Network::set_weight_decay, "设置权重衰减")
      .def("enable_debug_mode", &CNN::Network::enable_debug_mode,
           "启用调试模式", py::arg("enable") = true);
}