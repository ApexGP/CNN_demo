#include "cnn_core/tensor_core.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// 确保M_PI定义
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// 创建张量
cnn_core_status_t cnn_core_tensor_create(cnn_core_tensor_t *tensor,
                                         const size_t *dims, size_t ndim) {
    if (!tensor || !dims || ndim == 0 || ndim > CNN_CORE_MAX_DIMS) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 设置维度
    tensor->ndim = ndim;
    for (size_t i = 0; i < ndim; i++) {
        tensor->dims[i] = dims[i];
    }
    
    // 计算总元素数量
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= dims[i];
    }
    tensor->size = size;
    
    // 分配内存
    tensor->data = (float *)malloc(size * sizeof(float));
    if (!tensor->data) {
        return CNN_CORE_ERROR_MEMORY;
    }
    
    // 初始化为零
    memset(tensor->data, 0, size * sizeof(float));
    
    // 标记拥有数据所有权
    tensor->owns_data = 1;
    
    return CNN_CORE_SUCCESS;
}

// 销毁张量
cnn_core_status_t cnn_core_tensor_destroy(cnn_core_tensor_t *tensor) {
    if (!tensor) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 释放数据（如果拥有所有权）
    if (tensor->data && tensor->owns_data) {
        free(tensor->data);
        tensor->data = NULL;
    }
    
    return CNN_CORE_SUCCESS;
}

// 复制张量
cnn_core_status_t cnn_core_tensor_copy(cnn_core_tensor_t *dst,
                                       const cnn_core_tensor_t *src) {
    if (!dst || !src) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 如果目标已经有数据，先清理
    if (dst->data && dst->owns_data) {
        free(dst->data);
    }
    
    // 复制元数据
    dst->ndim = src->ndim;
    dst->size = src->size;
    for (size_t i = 0; i < src->ndim; i++) {
        dst->dims[i] = src->dims[i];
    }
    
    // 分配并复制数据
    dst->data = (float *)malloc(src->size * sizeof(float));
    if (!dst->data) {
        return CNN_CORE_ERROR_MEMORY;
    }
    
    memcpy(dst->data, src->data, src->size * sizeof(float));
    dst->owns_data = 1;
    
    return CNN_CORE_SUCCESS;
}

// 从内存创建张量视图
cnn_core_status_t cnn_core_tensor_view(cnn_core_tensor_t *tensor, float *data,
                                       const size_t *dims, size_t ndim) {
    if (!tensor || !data || !dims || ndim == 0 || ndim > CNN_CORE_MAX_DIMS) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 设置维度
    tensor->ndim = ndim;
    for (size_t i = 0; i < ndim; i++) {
        tensor->dims[i] = dims[i];
    }
    
    // 计算总元素数量
    size_t size = 1;
    for (size_t i = 0; i < ndim; i++) {
        size *= dims[i];
    }
    tensor->size = size;
    
    // 设置数据指针
    tensor->data = data;
    
    // 标记不拥有数据所有权
    tensor->owns_data = 0;
    
    return CNN_CORE_SUCCESS;
}

// 重塑张量
cnn_core_status_t cnn_core_tensor_reshape(cnn_core_tensor_t *tensor,
                                          const size_t *new_dims,
                                          size_t new_ndim) {
    if (!tensor || !new_dims || new_ndim == 0 || new_ndim > CNN_CORE_MAX_DIMS) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 计算新形状的元素总数
    size_t new_size = 1;
    for (size_t i = 0; i < new_ndim; i++) {
        new_size *= new_dims[i];
    }
    
    // 检查元素总数是否一致
    if (new_size != tensor->size) {
        return CNN_CORE_ERROR_SHAPE;
    }
    
    // 更新维度
    tensor->ndim = new_ndim;
    for (size_t i = 0; i < new_ndim; i++) {
        tensor->dims[i] = new_dims[i];
    }
    
    return CNN_CORE_SUCCESS;
}

// 填充张量
cnn_core_status_t cnn_core_tensor_fill(cnn_core_tensor_t *tensor, float value) {
    if (!tensor || !tensor->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = value;
    }
    
    return CNN_CORE_SUCCESS;
}

// 填充张量为零
cnn_core_status_t cnn_core_tensor_zero(cnn_core_tensor_t *tensor) {
    if (!tensor || !tensor->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    memset(tensor->data, 0, tensor->size * sizeof(float));
    
    return CNN_CORE_SUCCESS;
}

// 张量加法
cnn_core_status_t cnn_core_tensor_add(cnn_core_tensor_t *result,
                                      const cnn_core_tensor_t *a,
                                      const cnn_core_tensor_t *b) {
    if (!result || !a || !b || !result->data || !a->data || !b->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 检查形状兼容性
    if (a->ndim != b->ndim || a->ndim != result->ndim) {
        return CNN_CORE_ERROR_SHAPE;
    }
    
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->dims[i] != b->dims[i] || a->dims[i] != result->dims[i]) {
            return CNN_CORE_ERROR_SHAPE;
        }
    }
    
    // 执行加法
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
    
    return CNN_CORE_SUCCESS;
}

// 张量减法
cnn_core_status_t cnn_core_tensor_sub(cnn_core_tensor_t *result,
                                      const cnn_core_tensor_t *a,
                                      const cnn_core_tensor_t *b) {
    if (!result || !a || !b || !result->data || !a->data || !b->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 检查形状兼容性
    if (a->ndim != b->ndim || a->ndim != result->ndim) {
        return CNN_CORE_ERROR_SHAPE;
    }
    
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->dims[i] != b->dims[i] || a->dims[i] != result->dims[i]) {
            return CNN_CORE_ERROR_SHAPE;
        }
    }
    
    // 执行减法
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
    
    return CNN_CORE_SUCCESS;
}

// 张量元素乘法
cnn_core_status_t cnn_core_tensor_mul(cnn_core_tensor_t *result,
                                      const cnn_core_tensor_t *a,
                                      const cnn_core_tensor_t *b) {
    if (!result || !a || !b || !result->data || !a->data || !b->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 检查形状兼容性
    if (a->ndim != b->ndim || a->ndim != result->ndim) {
        return CNN_CORE_ERROR_SHAPE;
    }
    
    for (size_t i = 0; i < a->ndim; i++) {
        if (a->dims[i] != b->dims[i] || a->dims[i] != result->dims[i]) {
            return CNN_CORE_ERROR_SHAPE;
        }
    }
    
    // 执行元素乘法
    for (size_t i = 0; i < a->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
    
    return CNN_CORE_SUCCESS;
}

// 张量标量乘法
cnn_core_status_t cnn_core_tensor_scale(cnn_core_tensor_t *result,
                                        const cnn_core_tensor_t *tensor,
                                        float scalar) {
    if (!result || !tensor || !result->data || !tensor->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 检查形状兼容性
    if (tensor->ndim != result->ndim) {
        return CNN_CORE_ERROR_SHAPE;
    }
    
    for (size_t i = 0; i < tensor->ndim; i++) {
        if (tensor->dims[i] != result->dims[i]) {
            return CNN_CORE_ERROR_SHAPE;
        }
    }
    
    // 执行标量乘法
    for (size_t i = 0; i < tensor->size; i++) {
        result->data[i] = tensor->data[i] * scalar;
    }
    
    return CNN_CORE_SUCCESS;
}

// 张量矩阵乘法
cnn_core_status_t cnn_core_tensor_matmul(cnn_core_tensor_t *result,
                                         const cnn_core_tensor_t *a,
                                         const cnn_core_tensor_t *b) {
    if (!result || !a || !b || !result->data || !a->data || !b->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 检查是否为矩阵（2维张量）
    if (a->ndim != 2 || b->ndim != 2 || result->ndim != 2) {
        return CNN_CORE_ERROR_DIMENSION;
    }
    
    // 检查维度兼容性
    if (a->dims[1] != b->dims[0] || result->dims[0] != a->dims[0] || result->dims[1] != b->dims[1]) {
        return CNN_CORE_ERROR_SHAPE;
    }
    
    size_t m = a->dims[0];    // 结果行数
    size_t n = b->dims[1];    // 结果列数
    size_t k = a->dims[1];    // 内部维度
    
    // 简单实现的矩阵乘法（未优化）
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a->data[i * k + p] * b->data[p * n + j];
            }
            result->data[i * n + j] = sum;
        }
    }
    
    return CNN_CORE_SUCCESS;
}

// 张量转置
cnn_core_status_t cnn_core_tensor_transpose(cnn_core_tensor_t *result,
                                            const cnn_core_tensor_t *tensor) {
    if (!result || !tensor || !result->data || !tensor->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 当前仅支持2D张量转置
    if (tensor->ndim != 2 || result->ndim != 2) {
        return CNN_CORE_ERROR_DIMENSION;
    }
    
    // 检查结果形状是否为转置形状
    if (result->dims[0] != tensor->dims[1] || result->dims[1] != tensor->dims[0]) {
        return CNN_CORE_ERROR_SHAPE;
    }
    
    size_t rows = tensor->dims[0];
    size_t cols = tensor->dims[1];
    
    // 执行转置
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result->data[j * rows + i] = tensor->data[i * cols + j];
        }
    }
    
    return CNN_CORE_SUCCESS;
}

// 随机初始化张量
cnn_core_status_t cnn_core_tensor_rand(cnn_core_tensor_t *tensor, float min,
                                       float max, unsigned int seed) {
    if (!tensor || !tensor->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 设置随机数种子
    srand(seed);
    
    // 生成随机值
    for (size_t i = 0; i < tensor->size; i++) {
        float rand_01 = (float)rand() / RAND_MAX;  // 生成[0,1]之间的随机数
        tensor->data[i] = min + rand_01 * (max - min);  // 映射到[min,max]
    }
    
    return CNN_CORE_SUCCESS;
}

// 正态分布初始化张量
cnn_core_status_t cnn_core_tensor_randn(cnn_core_tensor_t *tensor, float mean,
                                        float stddev, unsigned int seed) {
    if (!tensor || !tensor->data) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 设置随机数种子
    srand(seed);
    
    // 使用Box-Muller变换生成正态分布
    for (size_t i = 0; i < tensor->size; i += 2) {
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        
        // 避免u1为0
        if (u1 < 1e-7f) u1 = 1e-7f;
        
        float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        tensor->data[i] = mean + stddev * z1;
        
        // 如果还有下一个元素
        if (i + 1 < tensor->size) {
            float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * (float)M_PI * u2);
            tensor->data[i + 1] = mean + stddev * z2;
        }
    }
    
    return CNN_CORE_SUCCESS;
}

// 获取张量元素
cnn_core_status_t cnn_core_tensor_get(const cnn_core_tensor_t *tensor,
                                      const size_t *indices, float *value) {
    if (!tensor || !tensor->data || !indices || !value) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 计算线性索引
    size_t linear_index = 0;
    size_t stride = 1;
    
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        if (indices[i] >= tensor->dims[i]) {
            return CNN_CORE_ERROR_INVALID_PARAM;  // 索引越界
        }
        linear_index += indices[i] * stride;
        stride *= tensor->dims[i];
    }
    
    if (linear_index >= tensor->size) {
        return CNN_CORE_ERROR_INVALID_PARAM;  // 索引越界
    }
    
    *value = tensor->data[linear_index];
    return CNN_CORE_SUCCESS;
}

// 设置张量元素
cnn_core_status_t cnn_core_tensor_set(cnn_core_tensor_t *tensor,
                                      const size_t *indices, float value) {
    if (!tensor || !tensor->data || !indices) {
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    // 计算线性索引
    size_t linear_index = 0;
    size_t stride = 1;
    
    for (int i = tensor->ndim - 1; i >= 0; i--) {
        if (indices[i] >= tensor->dims[i]) {
            return CNN_CORE_ERROR_INVALID_PARAM;  // 索引越界
        }
        linear_index += indices[i] * stride;
        stride *= tensor->dims[i];
    }
    
    if (linear_index >= tensor->size) {
        return CNN_CORE_ERROR_INVALID_PARAM;  // 索引越界
    }
    
    tensor->data[linear_index] = value;
    return CNN_CORE_SUCCESS;
}

// 打印张量
cnn_core_status_t cnn_core_tensor_print(const cnn_core_tensor_t *tensor) {
    if (!tensor) {
        printf("NULL tensor\n");
        return CNN_CORE_ERROR_INVALID_PARAM;
    }
    
    printf("Tensor shape: [");
    for (size_t i = 0; i < tensor->ndim; i++) {
        printf("%zu", tensor->dims[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf("], size: %zu\n", tensor->size);
    
    // 只打印较小的张量
    if (tensor->size <= 100 && tensor->data) {
        printf("Data: [");
        for (size_t i = 0; i < tensor->size; i++) {
            printf("%.4f", tensor->data[i]);
            if (i < tensor->size - 1) {
                printf(", ");
            }
        }
        printf("]\n");
    } else if (tensor->data) {
        printf("Data: (too large to display)\n");
    } else {
        printf("Data: NULL\n");
    }
    
    return CNN_CORE_SUCCESS;
} 