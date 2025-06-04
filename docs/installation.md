# CNN 框架安装指南

本文档提供 CNN 框架所需的所有依赖安装步骤，包括 C++库和 Python 包。

## 系统要求

- Windows 10/11（主要开发平台）
- 支持 C++17 的编译器（MinGW-w64 GCC 8.1+）
- CMake 3.15+
- Anaconda 或 Miniconda（Python 环境管理）
- 支持 CUDA 的 GPU（可选，用于 GPU 加速）

## 1. 安装 C++依赖

### 1.1 安装 MinGW-w64

1. 访问 [MSYS2 官网](https://www.msys2.org/) 下载并安装 MSYS2
2. 打开 MSYS2 MINGW64 终端，运行以下命令：

```bash
# 更新包数据库
pacman -Syu

# 安装MinGW-w64工具链
pacman -S --needed mingw-w64-x86_64-toolchain
pacman -S --needed mingw-w64-x86_64-cmake
pacman -S --needed mingw-w64-x86_64-make
```

3. 将 MinGW-w64 的 bin 目录添加到 PATH 环境变量，例如：
   - `C:\msys64\mingw64\bin`

### 1.2 安装 OpenBLAS

#### 选项 1：使用 MSYS2 安装（推荐）

```bash
pacman -S mingw-w64-x86_64-openblas
```

#### 选项 2：使用 vcpkg 安装

```bash
# 安装vcpkg
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.bat

# 安装OpenBLAS
vcpkg install openblas:x64-mingw-static
```

### 1.3 安装 CUDA（可选，用于 GPU 加速）

1. 访问 [NVIDIA CUDA 下载页面](https://developer.nvidia.com/cuda-downloads)
2. 选择 Windows 10/11 x86_64 平台，下载安装程序
3. 运行安装程序，按照向导完成安装
4. 安装完成后，确保 CUDA 路径已添加到环境变量

## 2. 安装 Python 依赖

### 2.1 创建并激活 Anaconda 环境

```bash
# 创建新环境
conda create -n cnn_env python=3.9

# 激活环境
conda activate cnn_env
```

### 2.2 安装 Python 依赖

使用项目根目录中的 requirements.txt 文件安装所有必要的 Python 包：

```bash
pip install -r requirements.txt
```

或者通过 conda 安装核心依赖（某些包可能需要 pip 安装）：

```bash
# 基础科学计算库
conda install numpy scipy

# pybind11 (C++绑定)
conda install -c conda-forge pybind11

# 可视化库
conda install matplotlib seaborn tensorboard plotly

# 图像处理和机器学习
conda install pillow scikit-learn scikit-image
conda install -c conda-forge opencv

# PyTorch (可选，用于参考)
conda install pytorch torchvision -c pytorch
```

## 3. 验证安装

### 3.1 验证 C++环境

在命令行中运行：

```bash
g++ --version
cmake --version
```

确保显示正确的版本信息。

### 3.2 验证 Python 环境

在 Python 中运行：

```python
import numpy
import pybind11
import matplotlib
import torch  # 如果已安装

print("环境验证完成！")
```

## 4. 构建 CNN 库

### 4.1 从源码构建

```bash
# 克隆仓库（如果尚未克隆）
git clone <repository-url>
cd <repository-directory>

# 创建构建目录
mkdir build
cd build

# 配置项目
cmake .. -G "MinGW Makefiles"

# 构建项目
cmake --build . -j4
```

### 4.2 构建选项

在`cmake`命令中可以使用以下选项：

- `-DUSE_CUDA=ON`: 启用 CUDA 支持（需要已安装 CUDA）
- `-DBUILD_PYTHON=ON`: 构建 Python 绑定
- `-DUSE_OPENBLAS=ON`: 使用 OpenBLAS 加速（需要已安装 OpenBLAS）
- `-DCMAKE_BUILD_TYPE=Debug`: 构建调试版本

示例：

```bash
cmake .. -G "MinGW Makefiles" -DUSE_OPENBLAS=ON -DBUILD_PYTHON=ON
```

## 5. 故障排除

### 5.1 找不到 OpenBLAS

如果 CMake 无法找到 OpenBLAS，可以手动指定路径：

```bash
cmake .. -G "MinGW Makefiles" -DOPENBLAS_ROOT=C:/path/to/openblas
```

### 5.2 找不到 CUDA

确保 CUDA 已正确安装，并且环境变量中包含 CUDA 路径。可以手动指定 CUDA 路径：

```bash
cmake .. -G "MinGW Makefiles" -DCUDA_TOOLKIT_ROOT_DIR=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.x
```

### 5.3 pybind11 问题

如果构建 Python 绑定时遇到问题，确保已正确安装 pybind11，并且 Python 路径已正确设置：

```bash
cmake .. -G "MinGW Makefiles" -DPYTHON_EXECUTABLE=$(which python)
```

## 6. 使用 Docker（可选）

为了确保开发环境一致，我们还提供了 Dockerfile：

```bash
# 构建Docker镜像
docker build -t cnn_framework .

# 运行容器
docker run -it --gpus all -v $(pwd):/workspace cnn_framework
```

## 下一步

安装完成后，请参阅以下文档继续：

- [快速开始指南](./quickstart.md)
- [API 参考](./api_reference.md)
- [架构设计](./architecture.md)
- [示例教程](./tutorials/README.md)
