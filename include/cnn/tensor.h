/**
 * @file tensor.h
 * @brief C++封装的张量类定义
 */

#ifndef CNN_TENSOR_H_
#define CNN_TENSOR_H_

#include "cnn_core/tensor_core.h"
#include <functional>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

namespace CNN {

/**
 * @brief 张量类，封装C核心层张量操作
 */
class Tensor {
public:
  /**
   * @brief 默认构造函数
   */
  Tensor();

  /**
   * @brief 从形状构造张量
   * @param dims 维度列表
   */
  Tensor(const std::vector<size_t> &dims);

  /**
   * @brief 从初始化列表构造张量
   * @param dims 维度初始化列表
   */
  Tensor(std::initializer_list<size_t> dims);

  /**
   * @brief 拷贝构造函数
   * @param other 另一个张量
   */
  Tensor(const Tensor &other);

  /**
   * @brief 移动构造函数
   * @param other 另一个张量
   */
  Tensor(Tensor &&other) noexcept;

  /**
   * @brief 从C核心张量构造
   * @param core_tensor C核心张量
   * @param take_ownership 是否接管所有权
   */
  explicit Tensor(const cnn_core_tensor_t &core_tensor,
                  bool take_ownership = false);

  /**
   * @brief 析构函数
   */
  ~Tensor();

  /**
   * @brief 拷贝赋值操作符
   * @param other 另一个张量
   * @return 张量引用
   */
  Tensor &operator=(const Tensor &other);

  /**
   * @brief 移动赋值操作符
   * @param other 另一个张量
   * @return 张量引用
   */
  Tensor &operator=(Tensor &&other) noexcept;

  /**
   * @brief 重塑张量
   * @param new_dims 新的维度列表
   * @return 张量引用
   */
  Tensor &reshape(const std::vector<size_t> &new_dims);

  /**
   * @brief 重塑张量
   * @param new_dims 新的维度初始化列表
   * @return 张量引用
   */
  Tensor &reshape(std::initializer_list<size_t> new_dims);

  /**
   * @brief 填充张量为常数
   * @param value 填充值
   * @return 张量引用
   */
  Tensor &fill(float value);

  /**
   * @brief 填充张量为零
   * @return 张量引用
   */
  Tensor &zeros();

  /**
   * @brief 填充张量为一
   * @return 张量引用
   */
  Tensor &ones();

  /**
   * @brief 随机初始化张量
   * @param min 最小值
   * @param max 最大值
   * @param seed 随机种子
   * @return 张量引用
   */
  Tensor &rand(float min = 0.0f, float max = 1.0f, unsigned int seed = 42);

  /**
   * @brief 正态分布初始化张量
   * @param mean 均值
   * @param stddev 标准差
   * @param seed 随机种子
   * @return 张量引用
   */
  Tensor &randn(float mean = 0.0f, float stddev = 1.0f, unsigned int seed = 42);

  /**
   * @brief Xavier均匀分布初始化
   * @param fan_in 输入单元数
   * @param fan_out 输出单元数
   * @return 张量引用
   */
  Tensor &xavier_uniform(size_t fan_in = 0, size_t fan_out = 0);

  /**
   * @brief He初始化
   * @param fan_in 输入单元数
   * @return 张量引用
   */
  Tensor &he_uniform(size_t fan_in = 0);

  /**
   * @brief 加法运算符
   * @param other 另一个张量
   * @return 结果张量
   */
  Tensor operator+(const Tensor &other) const;

  /**
   * @brief 加法赋值运算符
   * @param other 另一个张量
   * @return 张量引用
   */
  Tensor &operator+=(const Tensor &other);

  /**
   * @brief 减法运算符
   * @param other 另一个张量
   * @return 结果张量
   */
  Tensor operator-(const Tensor &other) const;

  /**
   * @brief 减法赋值运算符
   * @param other 另一个张量
   * @return 张量引用
   */
  Tensor &operator-=(const Tensor &other);

  /**
   * @brief 元素乘法运算符
   * @param other 另一个张量
   * @return 结果张量
   */
  Tensor operator*(const Tensor &other) const;

  /**
   * @brief 元素乘法赋值运算符
   * @param other 另一个张量
   * @return 张量引用
   */
  Tensor &operator*=(const Tensor &other);

  /**
   * @brief 标量乘法运算符
   * @param scalar 标量值
   * @return 结果张量
   */
  Tensor operator*(float scalar) const;

  /**
   * @brief 标量乘法赋值运算符
   * @param scalar 标量值
   * @return 张量引用
   */
  Tensor &operator*=(float scalar);

  /**
   * @brief 矩阵乘法
   * @param other 另一个张量
   * @return 结果张量
   */
  Tensor matmul(const Tensor &other) const;

  /**
   * @brief 转置
   * @return 结果张量
   */
  Tensor transpose() const;

  /**
   * @brief 获取张量形状
   * @return 形状向量
   */
  const std::vector<size_t> &shape() const;

  /**
   * @brief 获取张量维度数
   * @return 维度数
   */
  size_t ndim() const;

  /**
   * @brief 获取张量元素总数
   * @return 元素总数
   */
  size_t size() const;

  /**
   * @brief 获取元素
   * @param indices 索引列表
   * @return 元素值
   */
  float get(const std::vector<size_t> &indices) const;

  /**
   * @brief 设置元素
   * @param indices 索引列表
   * @param value 元素值
   */
  void set(const std::vector<size_t> &indices, float value);

  /**
   * @brief 获取一维索引元素
   * @param index 一维索引
   * @return 元素值
   */
  float operator[](size_t index) const;

  /**
   * @brief 获取一维索引元素的引用
   * @param index 一维索引
   * @return 元素引用
   */
  float &operator[](size_t index);

  /**
   * @brief 获取数据指针
   * @return 数据指针
   */
  float *data();

  /**
   * @brief 获取常量数据指针
   * @return 常量数据指针
   */
  const float *data() const;

  /**
   * @brief 获取C核心张量
   * @return C核心张量
   */
  const cnn_core_tensor_t &get_core_tensor() const;

  /**
   * @brief 打印张量
   */
  void print() const;

  /**
   * @brief 获取张量字符串表示
   * @return 字符串表示
   */
  std::string to_string() const;

  /**
   * @brief 应用函数到每个元素
   * @param func 应用函数
   * @return 新张量
   */
  Tensor apply(const std::function<float(float)> &func) const;

  /**
   * @brief 原地应用函数到每个元素
   * @param func 应用函数
   * @return 张量引用
   */
  Tensor &apply_inplace(const std::function<float(float)> &func);

  /**
   * @brief ReLU激活函数
   * @return 新张量
   */
  Tensor relu() const;

  /**
   * @brief 原地ReLU激活函数
   * @return 张量引用
   */
  Tensor &relu_inplace();

  /**
   * @brief Sigmoid激活函数
   * @return 新张量
   */
  Tensor sigmoid() const;

  /**
   * @brief 原地Sigmoid激活函数
   * @return 张量引用
   */
  Tensor &sigmoid_inplace();

  /**
   * @brief Tanh激活函数
   * @return 新张量
   */
  Tensor tanh() const;

  /**
   * @brief 原地Tanh激活函数
   * @return 张量引用
   */
  Tensor &tanh_inplace();

  /**
   * @brief Softmax激活函数
   * @return 新张量
   */
  Tensor softmax() const;

  /**
   * @brief 原地Softmax激活函数
   * @return 张量引用
   */
  Tensor &softmax_inplace();

  /**
   * @brief 克隆张量
   * @return 新张量
   */
  Tensor clone() const;

private:
  /**
   * @brief 核心张量数据
   */
  cnn_core_tensor_t core_tensor_;

  /**
   * @brief 形状缓存
   */
  std::vector<size_t> shape_;

  /**
   * @brief 更新形状缓存
   */
  void update_shape_cache();
};

// 全局标量乘法运算符
Tensor operator*(float scalar, const Tensor &tensor);

} // namespace CNN

#endif // CNN_TENSOR_H_