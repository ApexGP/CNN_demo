/**
 * @file tensor_binding.cpp
 * @brief 张量类的Python绑定
 */

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "cnn/tensor.h"

namespace py = pybind11;

// NumPy数组转CNN::Tensor
CNN::Tensor numpy_to_tensor(py::array_t<float> arr) {
  py::buffer_info buf = arr.request();

  // 获取维度
  std::vector<size_t> dims;
  for (size_t i = 0; i < buf.ndim; i++) {
    dims.push_back(static_cast<size_t>(buf.shape[i]));
  }

  // 创建张量
  CNN::Tensor tensor(dims);

  // 复制数据
  std::memcpy(tensor.data(), buf.ptr, sizeof(float) * tensor.size());

  return tensor;
}

// CNN::Tensor转NumPy数组
py::array_t<float> tensor_to_numpy(const CNN::Tensor &tensor) {
  // 获取维度
  const std::vector<size_t> &dims = tensor.shape();
  std::vector<py::ssize_t> shape(dims.begin(), dims.end());

  // 创建NumPy数组
  py::array_t<float> arr(shape);
  py::buffer_info buf = arr.request();

  // 复制数据
  std::memcpy(buf.ptr, tensor.data(), sizeof(float) * tensor.size());

  return arr;
}

void bind_tensor(py::module_ &m) {
  // 绑定张量类
  py::class_<CNN::Tensor>(m, "Tensor")
      // 构造函数
      .def(py::init<>())
      .def(py::init<const std::vector<size_t> &>())
      .def(py::init<std::initializer_list<size_t>>())
      .def(py::init(&numpy_to_tensor))

      // 属性
      .def_property_readonly("shape", &CNN::Tensor::shape)
      .def_property_readonly("ndim", &CNN::Tensor::ndim)
      .def_property_readonly("size", &CNN::Tensor::size)

      // 数据访问
      .def("__getitem__",
           [](const CNN::Tensor &t, size_t idx) {
             if (idx >= t.size())
               throw py::index_error("Index out of bounds");
             return t[idx];
           })
      .def("__setitem__",
           [](CNN::Tensor &t, size_t idx, float val) {
             if (idx >= t.size())
               throw py::index_error("Index out of bounds");
             t[idx] = val;
           })
      .def("get", &CNN::Tensor::get)
      .def("set", &CNN::Tensor::set)

      // 形状操作
      .def("reshape", static_cast<CNN::Tensor &(
                          CNN::Tensor::*)(const std::vector<size_t> &)>(
                          &CNN::Tensor::reshape))

      // 初始化
      .def("fill", &CNN::Tensor::fill)
      .def("zeros", &CNN::Tensor::zeros)
      .def("ones", &CNN::Tensor::ones)
      .def("rand", &CNN::Tensor::rand, py::arg("min") = 0.0f,
           py::arg("max") = 1.0f, py::arg("seed") = 42)
      .def("randn", &CNN::Tensor::randn, py::arg("mean") = 0.0f,
           py::arg("stddev") = 1.0f, py::arg("seed") = 42)
      .def("xavier_uniform", &CNN::Tensor::xavier_uniform,
           py::arg("fan_in") = 0, py::arg("fan_out") = 0)

      // 数学运算 - 使用显式绑定而不是操作符重载
      .def("__add__", &CNN::Tensor::operator+)
      .def("__iadd__", &CNN::Tensor::operator+=)
      .def("__sub__", &CNN::Tensor::operator-)
      .def("__isub__", &CNN::Tensor::operator-=)
      .def("__mul__",
           static_cast<CNN::Tensor (CNN::Tensor::*)(const CNN::Tensor &) const>(
               &CNN::Tensor::operator*))
      .def("__imul__",
           static_cast<CNN::Tensor &(CNN::Tensor::*)(const CNN::Tensor &)>(
               &CNN::Tensor::operator*=))
      .def("__mul__", static_cast<CNN::Tensor (CNN::Tensor::*)(float) const>(
                          &CNN::Tensor::operator*))
      .def("__imul__", static_cast<CNN::Tensor &(CNN::Tensor::*)(float)>(
                           &CNN::Tensor::operator*=))
      .def("__rmul__", [](const CNN::Tensor &t, float s) { return s * t; })

      // 矩阵运算
      .def("matmul", &CNN::Tensor::matmul)
      .def("transpose", &CNN::Tensor::transpose)

      // 激活函数
      .def("relu", &CNN::Tensor::relu)
      .def("relu_", &CNN::Tensor::relu_inplace)
      .def("sigmoid", &CNN::Tensor::sigmoid)
      .def("sigmoid_", &CNN::Tensor::sigmoid_inplace)
      .def("tanh", &CNN::Tensor::tanh)
      .def("tanh_", &CNN::Tensor::tanh_inplace)
      .def("softmax", &CNN::Tensor::softmax)
      .def("softmax_", &CNN::Tensor::softmax_inplace)

      // 应用函数
      .def("apply", &CNN::Tensor::apply)
      .def("apply_", &CNN::Tensor::apply_inplace)

      // 实用函数
      .def("print", &CNN::Tensor::print)
      .def("__repr__", &CNN::Tensor::to_string)
      .def("to_numpy", [](const CNN::Tensor &t) { return tensor_to_numpy(t); });

  // 辅助函数
  m.def("from_numpy", &numpy_to_tensor, "从NumPy数组创建张量");
  m.def("to_numpy", &tensor_to_numpy, "将张量转换为NumPy数组");
}