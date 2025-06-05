# ⚙️ 安装配置指南

快速设置 CNN 混合架构框架，轻松实现 90.9%准确率的深度学习模型。

## 📋 目录

- [系统要求](#️-系统要求)
- [快速安装](#-快速安装)
- [详细安装步骤](#-详细安装步骤)
- [依赖管理](#-依赖管理)
- [构建选项](#️-构建选项)
- [验证安装](#-验证安装)
- [环境变量配置](#-环境变量配置)
- [故障排除](#️-故障排除)

## 🖥️ 系统要求

### 最低要求

- **操作系统**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **编译器**: GCC 7+, Clang 6+, MSVC 2019+
- **内存**: 4GB RAM (推荐 8GB+)
- **磁盘空间**: 2GB 可用空间

### 推荐配置

- **CPU**: 4 核心以上，支持 AVX2 指令集
- **内存**: 8GB+ RAM
- **编译器**: GCC 9+, Clang 10+, MSVC 2022
- **Python**: 3.7+ (用于 Python 绑定)

## 🚀 快速安装

### Windows (推荐)

```powershell
# 1. 克隆项目
git clone https://github.com/yourusername/CNN_demo.git
cd CNN_demo

# 2. 一键构建（包含所有优化）
build.bat --release --with-openblas --with-python --run-tests

# 3. 运行演示
.\build\bin\mnist_training.exe
```

### Linux/macOS

```bash
# 1. 克隆项目
git clone https://github.com/yourusername/CNN_demo.git
cd CNN_demo

# 2. 一键构建
./build.sh --release --with-openblas --with-python --run-tests

# 3. 运行演示
./build/bin/mnist_training
```

### 使用 Docker (最简单)

```bash
# 1. 拉取预构建镜像
docker pull cnn-demo:latest

# 2. 运行容器
docker run -it cnn-demo:latest

# 3. 在容器内运行演示
./build/bin/mnist_training
```

## 🔧 详细安装步骤

### 步骤 1: 环境准备

#### Windows 环境

```powershell
# 选项A: 使用Visual Studio (推荐)
# 安装Visual Studio 2019/2022 with C++ workload
# 确保包含CMake和vcpkg

# 选项B: 使用MSYS2
# 下载并安装MSYS2: https://www.msys2.org/
pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-cmake
```

#### Linux 环境

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake git python3-dev

# CentOS/RHEL
sudo yum groupinstall "Development Tools"
sudo yum install cmake git python3-devel

# Arch Linux
sudo pacman -S base-devel cmake git python
```

#### macOS 环境

```bash
# 安装Xcode命令行工具
xcode-select --install

# 使用Homebrew安装依赖
brew install cmake git python@3.9

# 或使用MacPorts
sudo port install cmake git python39
```

### 步骤 2: 依赖安装

#### 使用 vcpkg (推荐)

```bash
# 1. 安装vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg

# Windows
.\bootstrap-vcpkg.bat
.\vcpkg integrate install

# Linux/macOS
./bootstrap-vcpkg.sh

# 2. 安装依赖包
vcpkg install openblas gtest pybind11

# 3. 设置环境变量 (Windows)
set CMAKE_TOOLCHAIN_FILE=[vcpkg root]/scripts/buildsystems/vcpkg.cmake
```

#### 使用系统包管理器

```bash
# Ubuntu/Debian
sudo apt install libopenblas-dev libgtest-dev python3-pybind11

# CentOS/RHEL
sudo yum install openblas-devel gtest-devel python3-pybind11

# macOS (Homebrew)
brew install openblas googletest pybind11

# Arch Linux
sudo pacman -S openblas gtest pybind11
```

#### 使用 Conda

```bash
# 创建虚拟环境
conda create -n cnn-demo python=3.8
conda activate cnn-demo

# 安装依赖
conda install -c conda-forge openblas gtest pybind11 numpy matplotlib
```

### 步骤 3: Python 环境设置

```bash
# 创建虚拟环境
python -m venv venv

# 激活环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate

# 安装Python依赖
pip install -r requirements.txt
```

### 步骤 4: 项目构建

#### 基础构建

```bash
# 创建构建目录
mkdir build && cd build

# 配置项目
cmake ..

# 编译
cmake --build . --config Release

# 安装 (可选)
cmake --install . --prefix install
```

#### 使用构建脚本 (推荐)

```bash
# 查看所有选项
./build.sh --help

# 常用构建组合
./build.sh --clean --release                    # 最基础版本
./build.sh --release --with-openblas           # 高性能版本
./build.sh --release --with-python             # Python集成版本
./build.sh --release --with-openblas --with-python --run-tests  # 完整版本
```

## 📦 依赖管理

### 核心依赖

| 依赖           | 版本要求   | 用途           | 必需性     |
| -------------- | ---------- | -------------- | ---------- |
| **CMake**      | 3.15+      | 构建系统       | 必需       |
| **C++编译器**  | C++17 支持 | 核心编译       | 必需       |
| **OpenBLAS**   | 0.3.0+     | 高性能数学运算 | 推荐       |
| **OpenMP**     | 4.0+       | 并行计算       | 推荐       |
| **pybind11**   | 2.6+       | Python 绑定    | 可选       |
| **GoogleTest** | 1.10+      | 单元测试       | 开发时推荐 |

### Python 依赖

```txt
# requirements.txt
numpy>=1.19.0
matplotlib>=3.3.0
pybind11>=2.6.0
pytest>=6.0.0        # 用于测试
jupyter>=1.0.0        # 用于示例notebook
```

### 检查依赖状态

```bash
# 运行依赖检查脚本
python scripts/check_dependencies.py

# 示例输出
✅ CMake 3.20.0 - OK
✅ GCC 9.4.0 - OK
✅ OpenBLAS 0.3.15 - OK
✅ OpenMP 4.5 - OK
✅ Python 3.8.10 - OK
⚠️  CUDA not found - GPU加速不可用
✅ 所有必需依赖已满足
```

## ⚙️ 构建选项

### CMake 配置选项

```cmake
# 基本选项
-DCMAKE_BUILD_TYPE=Release          # 构建类型: Debug|Release|RelWithDebInfo
-DCMAKE_INSTALL_PREFIX=./install    # 安装路径

# 功能选项
-DWITH_OPENBLAS=ON                  # 启用OpenBLAS高性能数学库
-DWITH_OPENMP=ON                    # 启用OpenMP并行计算
-DWITH_PYTHON=ON                    # 构建Python绑定
-DWITH_CUDA=OFF                     # 启用CUDA支持 (实验性)

# 测试和调试选项
-DBUILD_TESTS=ON                    # 构建单元测试
-DBUILD_EXAMPLES=ON                 # 构建示例程序
-DENABLE_COVERAGE=OFF               # 代码覆盖率分析
```

### 高级构建配置

```bash
# 自定义编译器
cmake .. -DCMAKE_CXX_COMPILER=g++-9

# 交叉编译 (ARM)
cmake .. -DCMAKE_TOOLCHAIN_FILE=arm-toolchain.cmake

# 静态链接
cmake .. -DBUILD_SHARED_LIBS=OFF

# 优化特定CPU
cmake .. -DCMAKE_CXX_FLAGS="-march=native -mtune=native"

# 调试构建
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=ON
```

### 性能调优构建

```bash
# 最大性能构建
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DWITH_OPENBLAS=ON \
  -DWITH_OPENMP=ON \
  -DCMAKE_CXX_FLAGS="-O3 -march=native -DNDEBUG" \
  -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON

# 内存优化构建 (低内存环境)
cmake .. \
  -DCMAKE_BUILD_TYPE=MinSizeRel \
  -DWITH_OPENBLAS=OFF \
  -DCMAKE_CXX_FLAGS="-Os -ffunction-sections -fdata-sections"
```

## ✅ 验证安装

### 运行测试套件

```bash
# 构建时运行测试
./build.sh --run-tests

# 手动运行测试
cd build
ctest --output-on-failure

# 运行特定测试
./bin/test_tensor
./bin/test_layers
./bin/test_network
```

### 运行示例程序

```bash
# C++示例
./build/bin/mnist_training

# Python示例 (如果构建了Python绑定)
cd examples/python_examples
python mnist_demo.py

# 基准测试
./build/bin/benchmark_conv
./build/bin/benchmark_fc
```

### 验证功能

```cpp
// 快速验证脚本 (verify_installation.cpp)
#include "cnn/network.h"
#include <iostream>

int main() {
    std::cout << "验证CNN框架安装..." << std::endl;

    // 创建简单网络
    CNN::Network network;
    network.add_conv_layer(8, 3);
    network.add_relu_layer();
    network.add_fc_layer(10);

    std::cout << "✅ 网络创建成功" << std::endl;
    std::cout << "✅ 参数数量: " << network.get_num_parameters() << std::endl;

    // 测试前向传播
    CNN::Tensor input({1, 28, 28});
    input.random_normal();
    auto output = network.forward(input);

    std::cout << "✅ 前向传播成功" << std::endl;
    std::cout << "✅ 输出形状: " << output.size() << std::endl;
    std::cout << "🎉 安装验证完成!" << std::endl;

    return 0;
}
```

## 📋 环境变量配置

### CNN_DEMO_ROOT 项目根目录

为了让程序能够从任何位置找到数据文件，建议设置`CNN_DEMO_ROOT`环境变量：

#### 自动设置(推荐)

```bash
# Windows
scripts\setup_env.bat

# Linux/macOS
source scripts/setup_env.sh
```

#### 手动设置

```bash
# Windows (PowerShell)
$env:CNN_DEMO_ROOT = "C:\path\to\CNN_demo"
# 或永久设置
setx CNN_DEMO_ROOT "C:\path\to\CNN_demo"

# Linux/macOS
export CNN_DEMO_ROOT="/path/to/CNN_demo"
# 添加到 ~/.bashrc 或 ~/.zshrc 使其永久生效
echo 'export CNN_DEMO_ROOT="/path/to/CNN_demo"' >> ~/.bashrc
```

#### 智能路径解析

即使不设置环境变量，程序也会自动尝试以下路径：

```
./data                    # 从项目根目录运行
../data                   # 从build目录运行
../../data                # 从build/bin目录运行
../../../data             # 从build/Debug/bin等深层目录运行
./CNN_demo/data           # 从上级目录运行
../CNN_demo/data          # 从兄弟目录运行
```

如果都找不到，程序会自动生成随机数据进行演示。

### 常用环境变量

```bash
# ~/.bashrc 或 ~/.zshrc
export CNN_DEMO_ROOT=/path/to/CNN_demo
export PATH=$CNN_DEMO_ROOT/build/bin:$PATH
export LD_LIBRARY_PATH=$CNN_DEMO_ROOT/build/lib:$LD_LIBRARY_PATH

# OpenMP配置
export OMP_NUM_THREADS=4              # 线程数
export OMP_SCHEDULE=dynamic           # 调度策略
export OMP_PROC_BIND=true             # 线程绑定

# OpenBLAS配置
export OPENBLAS_NUM_THREADS=1         # 防止过度并行化
export OPENBLAS_CORETYPE=Haswell      # 指定CPU类型

# Python路径
export PYTHONPATH=$CNN_DEMO_ROOT/build/python:$PYTHONPATH
```

### Windows 环境变量

```powershell
# PowerShell配置文件
$env:CNN_DEMO_ROOT = "C:\path\to\CNN_demo"
$env:PATH += ";$env:CNN_DEMO_ROOT\build\bin"

# 或使用系统设置
setx CNN_DEMO_ROOT "C:\path\to\CNN_demo"
setx PATH "%PATH%;%CNN_DEMO_ROOT%\build\bin"
```

## 🛠️ 故障排除

### 常见问题及解决方案

#### 问题 1: CMake 版本过低

```bash
# 错误信息
CMake Error: CMake 3.10 or higher is required. You are running version 3.5

# 解决方案
# Ubuntu
sudo apt remove cmake
sudo snap install cmake --classic

# 或从源码编译
wget https://github.com/Kitware/CMake/releases/download/v3.22.0/cmake-3.22.0.tar.gz
tar -xzf cmake-3.22.0.tar.gz && cd cmake-3.22.0
./bootstrap && make -j4 && sudo make install
```

#### 问题 2: 编译器不支持 C++17

```bash
# 错误信息
error: This file requires compiler and library support for the ISO C++ 2017 standard

# 解决方案
# 更新编译器
sudo apt install gcc-9 g++-9
export CC=gcc-9 CXX=g++-9

# 或指定CMake使用特定编译器
cmake .. -DCMAKE_CXX_COMPILER=g++-9
```

#### 问题 3: OpenBLAS 未找到

```bash
# 错误信息
Could NOT find OpenBLAS (missing: OpenBLAS_LIB OpenBLAS_INCLUDE_DIR)

# 解决方案选项A: 系统安装
sudo apt install libopenblas-dev

# 解决方案选项B: 手动指定路径
cmake .. -DOpenBLAS_ROOT=/usr/local/openblas

# 解决方案选项C: 禁用OpenBLAS
cmake .. -DWITH_OPENBLAS=OFF
```

#### 问题 4: Python 绑定编译失败

```bash
# 错误信息
Could NOT find pybind11 (missing: pybind11_DIR)

# 解决方案
pip install pybind11[global]
# 或
conda install -c conda-forge pybind11

# 手动指定路径
cmake .. -Dpybind11_DIR=$(python -m pybind11 --cmakedir)
```

#### 问题 5: 链接错误

```bash
# 错误信息
undefined reference to `openblas_xxx`

# 解决方案
# 确保库路径正确
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

# 或使用静态链接
cmake .. -DWITH_STATIC_LIBS=ON
```

#### 问题 6: Windows 特有问题

```powershell
# 问题: MSVC找不到
# 解决方案: 使用Developer Command Prompt
# 或设置环境变量
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"

# 问题: vcpkg集成失败
# 解决方案: 手动设置工具链
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:/vcpkg/scripts/buildsystems/vcpkg.cmake
```

### 性能问题诊断

#### 训练速度慢

```bash
# 检查编译优化
cmake .. -DCMAKE_BUILD_TYPE=Release

# 检查OpenBLAS状态
ldd ./build/bin/mnist_training | grep blas

# 检查OpenMP
export OMP_NUM_THREADS=4  # 设置线程数
export OMP_DISPLAY_ENV=TRUE  # 显示OpenMP信息
```

#### 内存占用过高

```bash
# 使用内存分析工具
valgrind --tool=massif ./build/bin/mnist_training

# 减少批大小
# 在代码中修改batch_size参数

# 使用内存优化构建
cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel
```

### 调试模式

```bash
# 调试构建
cmake .. -DCMAKE_BUILD_TYPE=Debug -DSANITIZE_ADDRESS=ON

# 使用调试器
gdb ./build/bin/mnist_training
# 或
lldb ./build/bin/mnist_training

# 内存检查
valgrind --leak-check=full ./build/bin/mnist_training
```

## 📋 环境变量配置

### 常用环境变量

```bash
# ~/.bashrc 或 ~/.zshrc
export CNN_DEMO_ROOT=/path/to/CNN_demo
export PATH=$CNN_DEMO_ROOT/build/bin:$PATH
export LD_LIBRARY_PATH=$CNN_DEMO_ROOT/build/lib:$LD_LIBRARY_PATH

# OpenMP配置
export OMP_NUM_THREADS=4              # 线程数
export OMP_SCHEDULE=dynamic           # 调度策略
export OMP_PROC_BIND=true             # 线程绑定

# OpenBLAS配置
export OPENBLAS_NUM_THREADS=1         # 防止过度并行化
export OPENBLAS_CORETYPE=Haswell      # 指定CPU类型

# Python路径
export PYTHONPATH=$CNN_DEMO_ROOT/build/python:$PYTHONPATH
```

### Windows 环境变量

```powershell
# PowerShell配置文件
$env:CNN_DEMO_ROOT = "C:\path\to\CNN_demo"
$env:PATH += ";$env:CNN_DEMO_ROOT\build\bin"

# 或使用系统设置
setx CNN_DEMO_ROOT "C:\path\to\CNN_demo"
setx PATH "%PATH%;%CNN_DEMO_ROOT%\build\bin"
```

## 🎯 下一步

安装完成后，您可以：

1. **运行演示**: `./build/bin/mnist_training` 体验 90.9%准确率
2. **查看示例**: 浏览 `examples/` 目录了解用法
3. **阅读 API**: 参考 [API_GUIDE.md](API_GUIDE.md) 学习接口
4. **深入架构**: 研读 [ARCHITECTURE.md](ARCHITECTURE.md) 了解实现

祝您使用愉快！🎉
