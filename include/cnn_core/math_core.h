#pragma once

#include "tensor_core.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// 张量级别操作
int cnn_add(const cnn_core_tensor_t *a, const cnn_core_tensor_t *b,
            cnn_core_tensor_t *result);
int cnn_multiply_scalar(const cnn_core_tensor_t *tensor, float scalar,
                        cnn_core_tensor_t *result);
int cnn_matmul(const cnn_core_tensor_t *a, const cnn_core_tensor_t *b,
               cnn_core_tensor_t *result);
int cnn_relu(const cnn_core_tensor_t *input, cnn_core_tensor_t *output);
int cnn_sigmoid(const cnn_core_tensor_t *input, cnn_core_tensor_t *output);
int cnn_softmax(const cnn_core_tensor_t *input, cnn_core_tensor_t *output,
                int dim);
float cnn_sum(const cnn_core_tensor_t *tensor);
float cnn_mean(const cnn_core_tensor_t *tensor);

#ifdef USE_OPENBLAS
int cnn_matmul_blas(const cnn_core_tensor_t *a, const cnn_core_tensor_t *b,
                    cnn_core_tensor_t *result);
#endif

// SIMD优化函数声明
void cnn_add_simd(const float *a, const float *b, float *result, int size);
void cnn_multiply_simd(const float *a, const float *b, float *result, int size);
void cnn_multiply_scalar_simd(const float *input, float scalar, float *result,
                              int size);

// 向量点积
float cnn_core_math_dot(const float *a, const float *b, size_t n);

// 向量范数
float cnn_core_math_norm(const float *a, size_t n, int p);

// 向量加法
void cnn_core_math_vector_add(float *result, const float *a, const float *b,
                              size_t n);

// 向量减法
void cnn_core_math_vector_sub(float *result, const float *a, const float *b,
                              size_t n);

// 向量乘法（逐元素）
void cnn_core_math_vector_mul(float *result, const float *a, const float *b,
                              size_t n);

// 向量除法（逐元素）
void cnn_core_math_vector_div(float *result, const float *a, const float *b,
                              size_t n);

// 向量缩放
void cnn_core_math_vector_scale(float *result, const float *a, float scalar,
                                size_t n);

// 矩阵乘法
void cnn_core_math_matmul(float *result, const float *a, const float *b,
                          size_t m, size_t n, size_t k);

// 矩阵转置
void cnn_core_math_transpose(float *result, const float *a, size_t rows,
                             size_t cols);

// 激活函数：ReLU
void cnn_core_math_relu(float *result, const float *a, size_t n);

// 激活函数：ReLU导数
void cnn_core_math_relu_grad(float *result, const float *a,
                             const float *grad_output, size_t n);

// 激活函数：Sigmoid
void cnn_core_math_sigmoid(float *result, const float *a, size_t n);

// 激活函数：Sigmoid导数
void cnn_core_math_sigmoid_grad(float *result, const float *sigmoid_output,
                                const float *grad_output, size_t n);

// 激活函数：Tanh
void cnn_core_math_tanh(float *result, const float *a, size_t n);

// 激活函数：Tanh导数
void cnn_core_math_tanh_grad(float *result, const float *tanh_output,
                             const float *grad_output, size_t n);

// 激活函数：Softmax
void cnn_core_math_softmax(float *result, const float *a, size_t n);

// 损失函数：均方误差(MSE)
float cnn_core_math_mse(const float *pred, const float *target, size_t n);

// 损失函数：MSE导数
void cnn_core_math_mse_grad(float *result, const float *pred,
                            const float *target, size_t n);

// 损失函数：交叉熵
float cnn_core_math_cross_entropy(const float *pred, const float *target,
                                  size_t n);

// 损失函数：交叉熵导数
void cnn_core_math_cross_entropy_grad(float *result, const float *pred,
                                      const float *target, size_t n);

// 随机数生成：均匀分布
void cnn_core_math_rand_uniform(float *result, size_t n, float min, float max,
                                unsigned int seed);

// 随机数生成：正态分布（使用Box-Muller变换）
void cnn_core_math_rand_normal(float *result, size_t n, float mean, float std,
                               unsigned int seed);

// 数学常量
#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// 内联数学函数
static inline float cnn_core_math_relu_single(float x) {
  return x > 0.0f ? x : 0.0f;
}

static inline float cnn_core_math_sigmoid_single(float x) {
  return 1.0f / (1.0f + expf(-x));
}

static inline float cnn_core_math_tanh_single(float x) { return tanhf(x); }

static inline float cnn_core_math_clamp(float x, float min_val, float max_val) {
  return x < min_val ? min_val : (x > max_val ? max_val : x);
}

#ifdef __cplusplus
}
#endif