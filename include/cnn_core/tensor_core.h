/**
 * @file tensor_core.h
 * @brief C语言核心张量操作
 */

#ifndef CNN_CORE_TENSOR_CORE_H_
#define CNN_CORE_TENSOR_CORE_H_

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief 定义错误码
 */
typedef enum {
  CNN_CORE_SUCCESS = 0,           /**< 操作成功 */
  CNN_CORE_ERROR_INVALID_PARAM,   /**< 无效参数 */
  CNN_CORE_ERROR_MEMORY,          /**< 内存错误 */
  CNN_CORE_ERROR_SHAPE,           /**< 形状不兼容 */
  CNN_CORE_ERROR_DIMENSION,       /**< 维度错误 */
  CNN_CORE_ERROR_NOT_IMPLEMENTED, /**< 功能未实现 */
  CNN_CORE_ERROR_OTHER            /**< 其他错误 */
} cnn_core_status_t;

/**
 * @brief 最大张量维度数量
 */
#define CNN_CORE_MAX_DIMS 4

/**
 * @brief 张量结构体
 */
typedef struct {
  float *data;                    /**< 数据指针 */
  size_t dims[CNN_CORE_MAX_DIMS]; /**< 每个维度的大小 */
  size_t ndim;                    /**< 维度数量 */
  size_t size;                    /**< 总元素数量 */
  int owns_data;                  /**< 是否拥有数据所有权 */
} cnn_core_tensor_t;

/**
 * @brief 创建张量
 *
 * @param tensor 张量指针
 * @param dims 维度数组
 * @param ndim 维度数量
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_create(cnn_core_tensor_t *tensor,
                                         const size_t *dims, size_t ndim);

/**
 * @brief 销毁张量
 *
 * @param tensor 张量指针
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_destroy(cnn_core_tensor_t *tensor);

/**
 * @brief 复制张量
 *
 * @param dst 目标张量
 * @param src 源张量
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_copy(cnn_core_tensor_t *dst,
                                       const cnn_core_tensor_t *src);

/**
 * @brief 从内存创建张量视图
 *
 * @param tensor 张量指针
 * @param data 数据指针
 * @param dims 维度数组
 * @param ndim 维度数量
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_view(cnn_core_tensor_t *tensor, float *data,
                                       const size_t *dims, size_t ndim);

/**
 * @brief 重塑张量
 *
 * @param tensor 张量指针
 * @param new_dims 新维度数组
 * @param new_ndim 新维度数量
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_reshape(cnn_core_tensor_t *tensor,
                                          const size_t *new_dims,
                                          size_t new_ndim);

/**
 * @brief 填充张量
 *
 * @param tensor 张量指针
 * @param value 填充值
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_fill(cnn_core_tensor_t *tensor, float value);

/**
 * @brief 填充张量为零
 *
 * @param tensor 张量指针
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_zero(cnn_core_tensor_t *tensor);

/**
 * @brief 张量加法
 *
 * @param result 结果张量
 * @param a 张量A
 * @param b 张量B
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_add(cnn_core_tensor_t *result,
                                      const cnn_core_tensor_t *a,
                                      const cnn_core_tensor_t *b);

/**
 * @brief 张量减法
 *
 * @param result 结果张量
 * @param a 张量A
 * @param b 张量B
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_sub(cnn_core_tensor_t *result,
                                      const cnn_core_tensor_t *a,
                                      const cnn_core_tensor_t *b);

/**
 * @brief 张量元素乘法
 *
 * @param result 结果张量
 * @param a 张量A
 * @param b 张量B
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_mul(cnn_core_tensor_t *result,
                                      const cnn_core_tensor_t *a,
                                      const cnn_core_tensor_t *b);

/**
 * @brief 张量标量乘法
 *
 * @param result 结果张量
 * @param tensor 张量
 * @param scalar 标量
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_scale(cnn_core_tensor_t *result,
                                        const cnn_core_tensor_t *tensor,
                                        float scalar);

/**
 * @brief 张量矩阵乘法
 *
 * @param result 结果张量
 * @param a 张量A
 * @param b 张量B
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_matmul(cnn_core_tensor_t *result,
                                         const cnn_core_tensor_t *a,
                                         const cnn_core_tensor_t *b);

/**
 * @brief 张量转置
 *
 * @param result 结果张量
 * @param tensor 输入张量
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_transpose(cnn_core_tensor_t *result,
                                            const cnn_core_tensor_t *tensor);

/**
 * @brief 随机初始化张量
 *
 * @param tensor 张量指针
 * @param min 最小值
 * @param max 最大值
 * @param seed 随机数种子
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_rand(cnn_core_tensor_t *tensor, float min,
                                       float max, unsigned int seed);

/**
 * @brief 正态分布初始化张量
 *
 * @param tensor 张量指针
 * @param mean 均值
 * @param stddev 标准差
 * @param seed 随机数种子
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_randn(cnn_core_tensor_t *tensor, float mean,
                                        float stddev, unsigned int seed);

/**
 * @brief 获取张量元素
 *
 * @param tensor 张量指针
 * @param indices 索引数组
 * @param value 值指针
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_get(const cnn_core_tensor_t *tensor,
                                      const size_t *indices, float *value);

/**
 * @brief 设置张量元素
 *
 * @param tensor 张量指针
 * @param indices 索引数组
 * @param value 值
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_set(cnn_core_tensor_t *tensor,
                                      const size_t *indices, float value);

/**
 * @brief 张量打印
 *
 * @param tensor 张量指针
 * @return cnn_core_status_t 状态码
 */
cnn_core_status_t cnn_core_tensor_print(const cnn_core_tensor_t *tensor);

#ifdef __cplusplus
}
#endif

#endif /* CNN_CORE_TENSOR_CORE_H_ */