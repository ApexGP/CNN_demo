#include "cnn_core/math_core.h"
#include <string.h>
#include <omp.h>
#include <math.h>
#include <stdlib.h>

#ifdef __AVX__
#include <immintrin.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

// 基础数学运算
int cnn_add(const cnn_core_tensor_t* a, const cnn_core_tensor_t* b, cnn_core_tensor_t* result) {
    if (!a || !b || !result) {
        return -1;
    }
    
    if (a->size != b->size || a->size != result->size) {
        return -1;
    }
    
    // 使用SIMD优化版本
    cnn_add_simd(a->data, b->data, result->data, a->size);
    return 0;
}

int cnn_multiply_scalar(const cnn_core_tensor_t* tensor, float scalar, cnn_core_tensor_t* result) {
    if (!tensor || !result) {
        return -1;
    }
    
    if (tensor->size != result->size) {
        return -1;
    }
    
    cnn_multiply_scalar_simd(tensor->data, scalar, result->data, tensor->size);
    return 0;
}

// 矩阵运算
int cnn_matmul(const cnn_core_tensor_t* a, const cnn_core_tensor_t* b, cnn_core_tensor_t* result) {
    if (!a || !b || !result) {
        return -1;
    }
    
    // 检查矩阵乘法的维度要求
    if (a->ndim != 2 || b->ndim != 2 || result->ndim != 2) {
        return -1;
    }
    
    int M = a->dims[0];
    int K = a->dims[1];
    int N = b->dims[1];
    
    if (b->dims[0] != K || result->dims[0] != M || result->dims[1] != N) {
        return -1;
    }

#ifdef USE_OPENBLAS
    return cnn_matmul_blas(a, b, result);
#else
    // 简单的矩阵乘法实现
    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            result->data[i * N + j] = sum;
        }
    }
    return 0;
#endif
}

// 激活函数
int cnn_relu(const cnn_core_tensor_t* input, cnn_core_tensor_t* output) {
    if (!input || !output) {
        return -1;
    }
    
    if (input->size != output->size) {
        return -1;
    }
    
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < input->size; i++) {
        output->data[i] = cnn_core_math_relu_single(input->data[i]);
    }
    
    return 0;
}

int cnn_sigmoid(const cnn_core_tensor_t* input, cnn_core_tensor_t* output) {
    if (!input || !output) {
        return -1;
    }
    
    if (input->size != output->size) {
        return -1;
    }
    
    size_t i;
    #pragma omp parallel for private(i)
    for (i = 0; i < input->size; i++) {
        output->data[i] = cnn_core_math_sigmoid_single(input->data[i]);
    }
    
    return 0;
}

int cnn_softmax(const cnn_core_tensor_t* input, cnn_core_tensor_t* output, int dim) {
    if (!input || !output) {
        return -1;
    }
    
    if (input->size != output->size) {
        return -1;
    }
    
    // 简化实现：假设是1D或最后一维的softmax
    int size = input->size;
    
    // 找到最大值以数值稳定
    float max_val = input->data[0];
    for (int i = 1; i < size; i++) {
        if (input->data[i] > max_val) {
            max_val = input->data[i];
        }
    }
    
    // 计算exp(x - max)
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output->data[i] = expf(input->data[i] - max_val);
        sum += output->data[i];
    }
    
    // 归一化
    for (int i = 0; i < size; i++) {
        output->data[i] /= sum;
    }
    
    return 0;
}

// SIMD优化版本
void cnn_add_simd(const float* a, const float* b, float* result, int size) {
#ifdef __AVX__
    // AVX优化版本 - 使用未对齐加载以提高兼容性
    int simd_size = (size / 8) * 8;
    
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < simd_size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // 处理剩余元素
    for (i = simd_size; i < size; i++) {
        result[i] = a[i] + b[i];
    }
#else
    // 标准版本 - 针对小数据集优化
    if (size < 1000) {
        // 小数据集不使用并行
        for (int i = 0; i < size; i++) {
            result[i] = a[i] + b[i];
        }
    } else {
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < size; i++) {
            result[i] = a[i] + b[i];
        }
    }
#endif
}

void cnn_multiply_simd(const float* a, const float* b, float* result, int size) {
#ifdef __AVX__
    int simd_size = (size / 8) * 8;
    
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < simd_size; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // 处理剩余元素
    for (i = simd_size; i < size; i++) {
        result[i] = a[i] * b[i];
    }
#else
    if (size < 1000) {
        for (int i = 0; i < size; i++) {
            result[i] = a[i] * b[i];
        }
    } else {
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < size; i++) {
            result[i] = a[i] * b[i];
        }
    }
#endif
}

void cnn_multiply_scalar_simd(const float* input, float scalar, float* result, int size) {
#ifdef __AVX__
    int simd_size = (size / 8) * 8;
    __m256 scalar_vec = _mm256_set1_ps(scalar);
    
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < simd_size; i += 8) {
        __m256 va = _mm256_loadu_ps(&input[i]);
        __m256 vr = _mm256_mul_ps(va, scalar_vec);
        _mm256_storeu_ps(&result[i], vr);
    }
    
    // 处理剩余元素
    for (i = simd_size; i < size; i++) {
        result[i] = input[i] * scalar;
    }
#else
    if (size < 1000) {
        for (int i = 0; i < size; i++) {
            result[i] = input[i] * scalar;
        }
    } else {
        int i;
        #pragma omp parallel for private(i)
        for (i = 0; i < size; i++) {
            result[i] = input[i] * scalar;
        }
    }
#endif
}

// 统计函数
float cnn_sum(const cnn_core_tensor_t* tensor) {
    if (!tensor) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    size_t i;
    #pragma omp parallel for private(i) reduction(+:sum)
    for (i = 0; i < tensor->size; i++) {
        sum += tensor->data[i];
    }
    
    return sum;
}

float cnn_mean(const cnn_core_tensor_t* tensor) {
    if (!tensor || tensor->size == 0) {
        return 0.0f;
    }
    
    return cnn_sum(tensor) / tensor->size;
}

#ifdef USE_OPENBLAS
int cnn_matmul_blas(const cnn_core_tensor_t* a, const cnn_core_tensor_t* b, cnn_core_tensor_t* result) {
    int M = a->dims[0];
    int K = a->dims[1];
    int N = b->dims[1];
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f, a->data, K,
                b->data, N,
                0.0f, result->data, N);
    
    return 0;
}
#endif

// 向量点积
float cnn_core_math_dot(const float *a, const float *b, size_t n) {
    if (!a || !b || n == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

// 向量范数
float cnn_core_math_norm(const float *a, size_t n, int p) {
    if (!a || n == 0) {
        return 0.0f;
    }
    
    if (p == 0) {
        // L0范数（非零元素个数）
        float count = 0.0f;
        for (size_t i = 0; i < n; i++) {
            if (a[i] != 0.0f) {
                count += 1.0f;
            }
        }
        return count;
    } else if (p == 1) {
        // L1范数（绝对值之和）
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            sum += fabsf(a[i]);
        }
        return sum;
    } else if (p == 2) {
        // L2范数（欧几里得范数）
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            sum += a[i] * a[i];
        }
        return sqrtf(sum);
    } else {
        // Lp范数
        float sum = 0.0f;
        for (size_t i = 0; i < n; i++) {
            sum += powf(fabsf(a[i]), (float)p);
        }
        return powf(sum, 1.0f / (float)p);
    }
}

// 向量加法
void cnn_core_math_vector_add(float *result, const float *a, const float *b, size_t n) {
    if (!result || !a || !b || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

// 向量减法
void cnn_core_math_vector_sub(float *result, const float *a, const float *b, size_t n) {
    if (!result || !a || !b || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] - b[i];
    }
}

// 向量乘法（逐元素）
void cnn_core_math_vector_mul(float *result, const float *a, const float *b, size_t n) {
    if (!result || !a || !b || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * b[i];
    }
}

// 向量除法（逐元素）
void cnn_core_math_vector_div(float *result, const float *a, const float *b, size_t n) {
    if (!result || !a || !b || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        if (b[i] != 0.0f) {
            result[i] = a[i] / b[i];
        } else {
            result[i] = 0.0f; // 处理除零
        }
    }
}

// 向量缩放
void cnn_core_math_vector_scale(float *result, const float *a, float scalar, size_t n) {
    if (!result || !a || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        result[i] = a[i] * scalar;
    }
}

// 矩阵乘法
void cnn_core_math_matmul(float *result, const float *a, const float *b, 
                          size_t m, size_t n, size_t k) {
    if (!result || !a || !b || m == 0 || n == 0 || k == 0) {
        return;
    }
    
    // C(m,n) = A(m,k) * B(k,n)
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += a[i * k + p] * b[p * n + j];
            }
            result[i * n + j] = sum;
        }
    }
}

// 矩阵转置
void cnn_core_math_transpose(float *result, const float *a, size_t rows, size_t cols) {
    if (!result || !a || rows == 0 || cols == 0) {
        return;
    }
    
    for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
            result[j * rows + i] = a[i * cols + j];
        }
    }
}

// 激活函数：ReLU
void cnn_core_math_relu(float *result, const float *a, size_t n) {
    if (!result || !a || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        result[i] = (a[i] > 0.0f) ? a[i] : 0.0f;
    }
}

// 激活函数：ReLU导数
void cnn_core_math_relu_grad(float *result, const float *a, const float *grad_output, size_t n) {
    if (!result || !a || !grad_output || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        result[i] = (a[i] > 0.0f) ? grad_output[i] : 0.0f;
    }
}

// 激活函数：Sigmoid
void cnn_core_math_sigmoid(float *result, const float *a, size_t n) {
    if (!result || !a || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        result[i] = 1.0f / (1.0f + expf(-a[i]));
    }
}

// 激活函数：Sigmoid导数
void cnn_core_math_sigmoid_grad(float *result, const float *sigmoid_output, const float *grad_output, size_t n) {
    if (!result || !sigmoid_output || !grad_output || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
        result[i] = grad_output[i] * sigmoid_output[i] * (1.0f - sigmoid_output[i]);
    }
}

// 激活函数：Tanh
void cnn_core_math_tanh(float *result, const float *a, size_t n) {
    if (!result || !a || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        result[i] = tanhf(a[i]);
    }
}

// 激活函数：Tanh导数
void cnn_core_math_tanh_grad(float *result, const float *tanh_output, const float *grad_output, size_t n) {
    if (!result || !tanh_output || !grad_output || n == 0) {
        return;
    }
    
    for (size_t i = 0; i < n; i++) {
        // tanh'(x) = 1 - tanh(x)^2
        result[i] = grad_output[i] * (1.0f - tanh_output[i] * tanh_output[i]);
    }
}

// 激活函数：Softmax
void cnn_core_math_softmax(float *result, const float *a, size_t n) {
    if (!result || !a || n == 0) {
        return;
    }
    
    // 找到最大值（避免数值溢出）
    float max_val = a[0];
    for (size_t i = 1; i < n; i++) {
        if (a[i] > max_val) {
            max_val = a[i];
        }
    }
    
    // 计算指数和总和
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        result[i] = expf(a[i] - max_val);
        sum += result[i];
    }
    
    // 归一化
    if (sum != 0.0f) {
        for (size_t i = 0; i < n; i++) {
            result[i] /= sum;
        }
    }
}

// 损失函数：均方误差(MSE)
float cnn_core_math_mse(const float *pred, const float *target, size_t n) {
    if (!pred || !target || n == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }
    
    return sum / (float)n;
}

// 损失函数：MSE导数
void cnn_core_math_mse_grad(float *result, const float *pred, const float *target, size_t n) {
    if (!result || !pred || !target || n == 0) {
        return;
    }
    
    float scale = 2.0f / (float)n;
    for (size_t i = 0; i < n; i++) {
        result[i] = scale * (pred[i] - target[i]);
    }
}

// 损失函数：交叉熵
float cnn_core_math_cross_entropy(const float *pred, const float *target, size_t n) {
    if (!pred || !target || n == 0) {
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        // 避免log(0)
        float p = fmaxf(pred[i], 1e-7f);
        p = fminf(p, 1.0f - 1e-7f);
        sum += target[i] * logf(p);
    }
    
    return -sum / (float)n;
}

// 损失函数：交叉熵导数
void cnn_core_math_cross_entropy_grad(float *result, const float *pred, const float *target, size_t n) {
    if (!result || !pred || !target || n == 0) {
        return;
    }
    
    float scale = -1.0f / (float)n;
    for (size_t i = 0; i < n; i++) {
        // 避免除零
        float p = fmaxf(pred[i], 1e-7f);
        p = fminf(p, 1.0f - 1e-7f);
        result[i] = scale * target[i] / p;
    }
}

// 随机数生成：均匀分布
void cnn_core_math_rand_uniform(float *result, size_t n, float min, float max, unsigned int seed) {
    if (!result || n == 0) {
        return;
    }
    
    // 设置随机数种子
    srand(seed);
    
    for (size_t i = 0; i < n; i++) {
        float rand_01 = (float)rand() / (float)RAND_MAX;  // [0,1]
        result[i] = min + rand_01 * (max - min);          // [min,max]
    }
}

// 随机数生成：正态分布（使用Box-Muller变换）
void cnn_core_math_rand_normal(float *result, size_t n, float mean, float std, unsigned int seed) {
    if (!result || n == 0) {
        return;
    }
    
    // 设置随机数种子
    srand(seed);
    
    for (size_t i = 0; i < n; i += 2) {
        float u1 = (float)rand() / (float)RAND_MAX;
        float u2 = (float)rand() / (float)RAND_MAX;
        
        // Box-Muller变换
        if (u1 < 1e-7f) u1 = 1e-7f;
        float z1 = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
        
        result[i] = mean + z1 * std;
        
        // 如果还有下一个元素
        if (i + 1 < n) {
            float z2 = sqrtf(-2.0f * logf(u1)) * sinf(2.0f * (float)M_PI * u2);
            result[i + 1] = mean + z2 * std;
        }
    }
} 