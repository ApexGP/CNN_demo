# 张量模块设计文档

## 概述

张量(Tensor)模块是 CNN 框架的核心数据结构，采用 C/C++混合架构设计，提供高效的多维数组操作和数学计算功能。

## 设计理念

### 1. 混合架构设计

```
┌─────────────────────────────────────────┐
│           C++ Tensor 封装层              │
│  ┌─────────────────────────────────────┐│
│  │        面向对象 API                  ││
│  │   - 构造函数、析构函数                 ││
│  │   - 运算符重载                        ││
│  │   - 异常处理                         ││
│  │   - RAII资源管理                     ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
┌─────────────────────────────────────────┐
│           C Core 计算层                  │
│  ┌─────────────────────────────────────┐│
│  │       高性能数值计算                  ││
│  │   - 内存管理                         ││
│  │   - 基础数学运算                      ││
│  │   - OpenMP并行化                     ││
│  │   - OpenBLAS集成                     ││
│  └─────────────────────────────────────┘│
└─────────────────────────────────────────┘
```

### 2. 核心特性

- **零拷贝设计**: C++层直接使用 C 层的内存，避免不必要的数据拷贝
- **自动内存管理**: 通过 RAII 确保内存安全
- **高性能计算**: 集成 OpenBLAS 和 OpenMP 加速
- **类型安全**: C++层提供类型检查和异常处理

## 模块结构

### C 核心层 (`cnn_core_tensor_t`)

**文件位置**: `include/cnn_core/tensor_core.h`, `src/core_c/tensor_core.c`

```c
typedef struct {
    float *data;           // 数据指针
    size_t *dims;          // 维度数组
    size_t ndim;           // 维度数量
    size_t size;           // 总元素数量
    int owns_data;         // 数据所有权标志
} cnn_core_tensor_t;
```

**核心函数**:

```c
// 创建和销毁
int cnn_core_tensor_create(cnn_core_tensor_t *tensor, const size_t *dims, size_t ndim);
void cnn_core_tensor_destroy(cnn_core_tensor_t *tensor);

// 基础操作
int cnn_core_tensor_reshape(cnn_core_tensor_t *tensor, const size_t *new_dims, size_t new_ndim);
void cnn_core_tensor_fill(cnn_core_tensor_t *tensor, float value);

// 数学运算
int cnn_core_tensor_add(cnn_core_tensor_t *result, const cnn_core_tensor_t *a, const cnn_core_tensor_t *b);
int cnn_core_tensor_mul(cnn_core_tensor_t *result, const cnn_core_tensor_t *a, const cnn_core_tensor_t *b);
int cnn_core_tensor_matmul(cnn_core_tensor_t *result, const cnn_core_tensor_t *a, const cnn_core_tensor_t *b);
```

### C++封装层 (`CNN::Tensor`)

**文件位置**: `include/cnn/tensor.h`, `src/cpp/tensor.cpp`

```cpp
namespace CNN {
class Tensor {
private:
    cnn_core_tensor_t core_tensor_;  // C核心张量
    std::vector<size_t> shape_;      // 形状缓存

public:
    // 构造函数
    Tensor();
    Tensor(const std::vector<size_t>& dims);
    Tensor(std::initializer_list<size_t> dims);
    Tensor(const Tensor& other);
    Tensor(Tensor&& other) noexcept;

    // 运算符重载
    Tensor operator+(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator*(float scalar) const;
    float& operator[](size_t index);
    const float& operator[](size_t index) const;

    // 数据访问
    float* data();
    const float* data() const;
    size_t size() const;
    const std::vector<size_t>& shape() const;
};
}
```

## 依赖关系

### 第三方依赖

1. **OpenBLAS** (可选)

   - **用途**: 高性能矩阵运算加速
   - **影响**: 矩阵乘法性能提升 10-50 倍
   - **缺失处理**: 使用`NO_OPENBLAS`宏，回退到标准实现

2. **OpenMP** (可选)
   - **用途**: 多线程并行计算
   - **影响**: 大规模张量运算性能提升 2-4 倍
   - **缺失处理**: 单线程执行，功能不受影响

### 内部依赖

```
CNN::Tensor
    ↓
cnn_core_tensor_t
    ↓
math_core (数学运算)
    ↓
OpenBLAS (可选) + OpenMP (可选)
```

## 性能优化

### 1. 内存管理优化

- **内存对齐**: 确保数据按 cache line 对齐
- **预分配策略**: 减少频繁的内存分配/释放
- **引用计数**: 支持浅拷贝，减少不必要的数据复制

### 2. 计算优化

- **向量化**: 利用 SIMD 指令集
- **并行化**: OpenMP 并行循环
- **矩阵运算**: 调用优化的 BLAS 库

### 3. 缓存优化

- **数据局部性**: 优化内存访问模式
- **分块算法**: 大矩阵分块处理
- **预取**: 关键循环中的数据预取

## 使用示例

### 基础操作

```cpp
#include "cnn/tensor.h"

// 创建张量
CNN::Tensor a({2, 3, 4});  // 2×3×4张量
CNN::Tensor b = CNN::Tensor({2, 3, 4}).fill(1.0f);

// 数学运算
CNN::Tensor c = a + b;     // 逐元素加法
CNN::Tensor d = a * 2.0f;  // 标量乘法

// 矩阵运算
CNN::Tensor mat1({3, 4});
CNN::Tensor mat2({4, 5});
CNN::Tensor result = mat1.matmul(mat2);  // 矩阵乘法

// 激活函数
CNN::Tensor activated = a.relu();
CNN::Tensor normalized = a.softmax();
```

### 高级操作

```cpp
// 随机初始化
CNN::Tensor weights({256, 128});
weights.rand(-0.1f, 0.1f, 42);  // 均匀分布随机初始化

// 形状变换
CNN::Tensor reshaped = weights.reshape({128, 256});
CNN::Tensor cloned = weights.clone();

// 数据访问
float* raw_data = weights.data();
const auto& shape = weights.shape();
size_t total_size = weights.size();
```

## 错误处理

### 异常类型

1. **形状不匹配**: `std::runtime_error`
2. **内存分配失败**: `std::bad_alloc`
3. **索引越界**: `std::out_of_range`
4. **无效参数**: `std::invalid_argument`

### 错误处理策略

```cpp
try {
    CNN::Tensor a({1000, 1000});
    CNN::Tensor b({1000, 500});
    CNN::Tensor c = a + b;  // 形状不匹配，抛出异常
} catch (const std::runtime_error& e) {
    std::cerr << "张量运算错误: " << e.what() << std::endl;
}
```

## 编译配置

### CMake 选项

```cmake
# 启用OpenBLAS支持
option(USE_OPENBLAS "使用OpenBLAS加速" ON)

# 启用OpenMP支持
option(USE_OPENMP "使用OpenMP加速" ON)

# 编译时优化
if(USE_OPENBLAS)
    add_definitions(-DUSE_OPENBLAS)
endif()

if(NOT BLAS_FOUND)
    add_definitions(-DNO_OPENBLAS)
endif()
```

### 预处理器宏

- `USE_OPENBLAS`: 启用 OpenBLAS 支持
- `NO_OPENBLAS`: 禁用 OpenBLAS，使用标准实现
- `_OPENMP`: OpenMP 支持（编译器自动定义）

## 测试与验证

### 单元测试

```cpp
// 张量创建测试
TEST(TensorTest, Creation) {
    CNN::Tensor t({2, 3});
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

// 数学运算测试
TEST(TensorTest, Addition) {
    CNN::Tensor a({2, 2});
    CNN::Tensor b({2, 2});
    a.fill(1.0f);
    b.fill(2.0f);

    CNN::Tensor c = a + b;
    for(size_t i = 0; i < c.size(); ++i) {
        EXPECT_FLOAT_EQ(c[i], 3.0f);
    }
}
```

### 性能基准

```cpp
// 矩阵乘法性能测试
void benchmark_matmul() {
    CNN::Tensor a({1000, 1000});
    CNN::Tensor b({1000, 1000});
    a.rand(-1.0f, 1.0f);
    b.rand(-1.0f, 1.0f);

    auto start = std::chrono::high_resolution_clock::now();
    CNN::Tensor c = a.matmul(b);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "矩阵乘法耗时: " << duration.count() << "ms" << std::endl;
}
```

## 未来改进

1. **GPU 支持**: 添加 CUDA 后端
2. **更多数据类型**: 支持 int8、fp16 等
3. **分布式计算**: 多机多卡训练支持
4. **内存优化**: 更智能的内存池管理
5. **JIT 编译**: 运行时代码生成优化
