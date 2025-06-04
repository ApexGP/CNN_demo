/**
 * @file layers_binding.cpp
 * @brief 层类的Python绑定实现
 */

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cnn/layers.h"

namespace py = pybind11;

void bind_layers(py::module_ &m) {
  // 绑定基础层类
  py::class_<CNN::Layer>(m, "Layer")
      .def("forward", &CNN::Layer::forward, "前向传播")
      .def("backward", &CNN::Layer::backward, "反向传播")
      .def("name", &CNN::Layer::name, "获取层名称")
      .def("output_shape", &CNN::Layer::output_shape, "计算输出形状")
      .def("train", &CNN::Layer::train, "设置训练模式", py::arg("mode") = true)
      .def("is_training", &CNN::Layer::is_training, "检查是否为训练模式");

  // 绑定卷积层
  py::class_<CNN::ConvLayer, CNN::Layer>(m, "ConvLayer")
      .def(py::init<int, int, int, int, bool>(), "创建卷积层",
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0,
           py::arg("bias") = true)
      .def(py::init<int, int, int, int, int, bool>(),
           "创建卷积层（指定输入通道数）", py::arg("in_channels"),
           py::arg("out_channels"), py::arg("kernel_size"),
           py::arg("stride") = 1, py::arg("padding") = 0,
           py::arg("bias") = true)
      .def("set_padding", &CNN::ConvLayer::set_padding, "设置填充")
      .def("set_stride", &CNN::ConvLayer::set_stride, "设置步长");

  // 绑定全连接层
  py::class_<CNN::FullyConnectedLayer, CNN::Layer>(m, "FullyConnectedLayer")
      .def(py::init<int, bool>(), "创建全连接层", py::arg("out_features"),
           py::arg("bias") = true)
      .def(py::init<int, int, bool>(), "创建全连接层（指定输入特征数）",
           py::arg("in_features"), py::arg("out_features"),
           py::arg("bias") = true);

  // 绑定激活函数层
  py::class_<CNN::ReLULayer, CNN::Layer>(m, "ReLULayer")
      .def(py::init<>(), "创建ReLU层");

  py::class_<CNN::SigmoidLayer, CNN::Layer>(m, "SigmoidLayer")
      .def(py::init<>(), "创建Sigmoid层");

  py::class_<CNN::TanhLayer, CNN::Layer>(m, "TanhLayer")
      .def(py::init<>(), "创建Tanh层");

  py::class_<CNN::SoftmaxLayer, CNN::Layer>(m, "SoftmaxLayer")
      .def(py::init<int>(), "创建Softmax层", py::arg("dim") = -1);

  // 绑定池化层
  py::class_<CNN::MaxPoolLayer, CNN::Layer>(m, "MaxPoolLayer")
      .def(py::init<int, int, int>(), "创建最大池化层", py::arg("kernel_size"),
           py::arg("stride") = -1, py::arg("padding") = 0);

  py::class_<CNN::AvgPoolLayer, CNN::Layer>(m, "AvgPoolLayer")
      .def(py::init<int, int, int>(), "创建平均池化层", py::arg("kernel_size"),
           py::arg("stride") = -1, py::arg("padding") = 0);

  // 绑定其他层
  py::class_<CNN::DropoutLayer, CNN::Layer>(m, "DropoutLayer")
      .def(py::init<float>(), "创建Dropout层", py::arg("p") = 0.5f);

  py::class_<CNN::FlattenLayer, CNN::Layer>(m, "FlattenLayer")
      .def(py::init<>(), "创建展平层");

  py::class_<CNN::BatchNormLayer, CNN::Layer>(m, "BatchNormLayer")
      .def(py::init<int, float, float>(), "创建批标准化层",
           py::arg("num_features"), py::arg("eps") = 1e-5f,
           py::arg("momentum") = 0.1f);
}