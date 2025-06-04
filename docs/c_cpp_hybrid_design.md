# C/C++混合架构设计文档

## 架构概述

本 CNN 框架采用**C 语言核心 + C++封装**的混合架构设计，实现了高性能计算与现代编程接口的完美结合。

### 架构层次

```
Python API (pybind11)
        ↓
C++ 封装层 (面向对象接口)
        ↓
C 计算核心 (高性能实现)
        ↓
硬件优化层 (SIMD/CUDA/OpenBLAS)
```

## 核心优势

### 1. 性能优势

- **C 语言核心**: 极致的数值计算性能
- **内存对齐**: 32 字节对齐的内存分配
- **SIMD 优化**: AVX/SSE 指令集加速
- **OpenMP 并行化**: 多核 CPU 利用率最大化
- **OpenBLAS 集成**: 高度优化的线性代数库

### 2. 开发体验

- **现代 C++接口**: RAII、异常安全、移动语义
- **操作符重载**: 直观的数学运算表达
- **类型安全**: 强类型系统减少错误
- **自动内存管理**: 智能指针和析构函数

### 3. 扩展性

- **跨平台支持**: Windows/Linux/macOS
- **模块化设计**: 独立的 C 核心和 C++封装
- **插件式架构**: 可选的 CUDA/OpenBLAS 支持
- **Python 绑定**: 无缝的 Python 集成

## 目录结构

```
CNN_demo/
├── include/
│   ├── cnn_core/          # C语言核心头文件
│   │   ├── tensor_core.h  # 张量核心
│   │   ├── math_core.h    # 数学运算
│   │   └── conv_core.h    # 卷积运算
│   └── cnn/               # C++封装头文件
│       ├── tensor.h       # 张量类
│       ├── layers.h       # 网络层
│       └── network.h      # 网络类
├── src/
│   ├── core_c/            # C语言核心实现
│   │   ├── tensor_core.c  # 张量核心实现
│   │   ├── math_core.c    # 数学运算实现
│   │   └── conv_core.c    # 卷积运算实现
│   └── cpp/               # C++封装实现
│       ├── tensor.cpp     # 张量类实现
│       ├── layers.cpp     # 网络层实现
│       └── network.cpp    # 网络类实现
└── examples/
    └── hybrid_test.cpp    # 混合架构演示
```

## C 语言核心设计

### 张量核心 (tensor_core.h/c)

#### 基础数据结构

```c
typedef struct {
    float* data;           // 数据指针（CPU）
    int* shape;            // 形状数组
    int* strides;          // 步长数组
    int ndim;              // 维度数
    int size;              // 总元素数
    cnn_device_t device;   // 设备类型
    cnn_dtype_t dtype;     // 数据类型

#ifdef USE_CUDA
    float* gpu_data;       // GPU数据指针
    int gpu_data_valid;    // GPU数据有效标志
    int cpu_data_valid;    // CPU数据有效标志
#endif
} cnn_tensor_t;
```

#### 核心功能

- **内存管理**: 对齐内存分配、自动释放
- **数据访问**: 多维索引、平坦索引
- **形状操作**: reshape、transpose、squeeze
- **初始化方法**: zeros、ones、random、Xavier 等

### 数学运算核心 (math_core.h/c)

#### 基础运算

```c
// 张量运算
int cnn_add(const cnn_tensor_t* a, const cnn_tensor_t* b, cnn_tensor_t* result);
int cnn_multiply_scalar(const cnn_tensor_t* tensor, float scalar, cnn_tensor_t* result);
int cnn_matmul(const cnn_tensor_t* a, const cnn_tensor_t* b, cnn_tensor_t* result);

// 激活函数
int cnn_relu(const cnn_tensor_t* input, cnn_tensor_t* output);
int cnn_sigmoid(const cnn_tensor_t* input, cnn_tensor_t* output);
int cnn_softmax(const cnn_tensor_t* input, cnn_tensor_t* output, int dim);
```

#### SIMD 优化

```c
// AVX优化版本
void cnn_add_simd(const float* a, const float* b, float* result, int size);
void cnn_multiply_scalar_simd(const float* input, float scalar, float* result, int size);
```

## C++封装设计

### 张量类 (tensor.h/cpp)

#### 面向对象接口

```cpp
namespace CNN {
    class Tensor {
    private:
        cnn_tensor_t* core_;  // C语言核心

    public:
        // 现代C++特性
        Tensor(const std::vector<int>& shape);
        ~Tensor();
        Tensor(const Tensor& other);
        Tensor(Tensor&& other) noexcept;

        // 直观的操作符
        Tensor operator+(const Tensor& other) const;
        Tensor operator*(float scalar) const;
        Tensor matmul(const Tensor& other) const;

        // 链式调用
        Tensor relu() const;
        Tensor sigmoid() const;
        Tensor softmax(int dim = -1) const;
    };
}
```

#### 关键特性

- **RAII 管理**: 自动内存管理
- **异常安全**: 错误处理机制
- **移动语义**: 高效的对象传递
- **操作符重载**: 直观的数学表达

## 构建系统

### CMake 配置

```cmake
# C语言核心库
add_library(cnn_core_c STATIC ${C_CORE_SOURCES})
set_target_properties(cnn_core_c PROPERTIES C_STANDARD 11)

# C++封装库
add_library(cnn_core_cpp STATIC ${CPP_SOURCES})
set_target_properties(cnn_core_cpp PROPERTIES CXX_STANDARD 17)
target_link_libraries(cnn_core_cpp PUBLIC cnn_core_c)

# 最终库
add_library(cnn_core INTERFACE)
target_link_libraries(cnn_core INTERFACE cnn_core_cpp)
```

### 编译选项

- **C 核心**: `-O3 -march=native -fopenmp`
- **C++封装**: `-std=c++17 -Wall -Wextra`
- **平台适配**: Windows/Linux/macOS 兼容

## 性能特性

### 内存优化

- **对齐分配**: 32 字节对齐提升 SIMD 性能
- **连续内存**: C 风格数组布局
- **零拷贝**: 高效的数据传递

### 计算优化

- **SIMD 指令**: AVX/SSE 向量化计算
- **OpenMP 并行**: 多线程计算加速
- **OpenBLAS**: 高度优化的 BLAS 库
- **GPU 支持**: CUDA 加速（可选）

### 编译器优化

- **内联函数**: 激活函数内联
- **循环展开**: 编译器自动优化
- **分支预测**: 条件代码优化

## 使用示例

### C++接口使用

```cpp
#include "cnn/tensor.h"

using namespace CNN;

int main() {
    // 创建张量
    Tensor a({2, 3});
    Tensor b({2, 3});

    // 初始化
    a.uniform(0.0f, 1.0f);
    b.ones();

    // 数学运算
    auto c = a + b;
    auto d = c.relu();

    // 矩阵乘法
    Tensor weight({3, 4});
    weight.xavier_uniform();
    auto output = a.matmul(weight);

    // 激活函数
    auto activated = output.relu().softmax();

    activated.print();
    return 0;
}
```

### Python 接口使用

```python
import cnn_py

# 创建张量
a = cnn_py.Tensor([2, 3])
b = cnn_py.Tensor([2, 3])

# 数学运算
c = a + b
d = c.relu()

# 打印结果
d.print()
```

## 扩展指南

### 添加新的数学运算

1. 在`math_core.h`中声明 C 函数
2. 在`math_core.c`中实现 C 函数
3. 在`tensor.h`中添加 C++方法
4. 在`tensor.cpp`中实现 C++封装

### 添加新的层类型

1. 在`conv_core.h`中声明 C 函数
2. 在`conv_core.c`中实现核心算法
3. 在`layers.h`中添加 C++类
4. 在`layers.cpp`中实现封装

### CUDA 支持扩展

1. 创建`.cu`文件实现 CUDA 核函数
2. 在 CMakeLists.txt 中添加 CUDA 源文件
3. 使用`#ifdef USE_CUDA`条件编译

## 测试验证

混合架构测试程序验证了以下特性：

- ✅ C 语言核心的高性能计算
- ✅ C++封装的现代接口
- ✅ RAII 自动内存管理
- ✅ 异常安全的错误处理
- ✅ 直观的操作符重载

运行测试：

```bash
cd build
./hybrid_test
```

## 总结

C/C++混合架构实现了：

1. **极致性能**: C 语言核心提供最优计算性能
2. **开发友好**: C++封装提供现代编程体验
3. **高度优化**: SIMD、OpenMP、BLAS 多重加速
4. **易于扩展**: 模块化设计支持功能扩展
5. **跨平台**: 统一的构建和部署方案

这种架构充分发挥了 C 和 C++各自的优势，为 CNN 框架提供了坚实的技术基础。
