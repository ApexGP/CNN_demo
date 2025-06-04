# CNN 混合架构框架依赖指南

## 概述

本文档详细说明 CNN 混合架构框架的所有依赖项，提供完整的安装方法和故障排除指南。阅读本指南可以避免编译失败问题。

## 依赖分类

### 必需依赖（REQUIRED）

这些依赖是框架正常编译的最低要求：

| 依赖项        | 最低版本   | 用途         | 获取方式                                 |
| ------------- | ---------- | ------------ | ---------------------------------------- |
| **CMake**     | 3.15+      | 构建系统     | [cmake.org](https://cmake.org/download/) |
| **C++编译器** | C++17 支持 | 编译主框架   | 见下方编译器安装                         |
| **C 编译器**  | C99 支持   | 编译核心模块 | 随 C++编译器提供                         |

### 可选依赖（OPTIONAL）

这些依赖会显著提升性能，但缺失时框架仍可正常工作：

| 依赖项           | 版本要求 | 性能提升         | 缺失后果               | 优先级 |
| ---------------- | -------- | ---------------- | ---------------------- | ------ |
| **OpenBLAS**     | 0.3.0+   | 矩阵运算 10-50x  | 使用标准实现，性能较低 | 🔥 高  |
| **OpenMP**       | 任意     | 多线程 2-4x      | 单线程执行             | 🔥 高  |
| **Python**       | 3.7+     | 启用 Python 接口 | Python 绑定不可用      | 🔶 中  |
| **pybind11**     | 2.6+     | Python-C++桥接   | Python 绑定不可用      | 🔶 中  |
| **CUDA Toolkit** | 10.0+    | GPU 加速 100x+   | GPU 加速不可用         | 🔵 低  |

## 平台特定安装指南

## Windows 平台

### 方法 1：自动安装（推荐）

使用提供的安装脚本：

```powershell
# 确保已安装Anaconda/Miniconda
# 下载地址：https://www.anaconda.com/products/distribution

# 运行自动安装脚本
install_env.bat
```

**脚本执行内容**：

1. 创建 cnn_demo Conda 环境 (Python 3.8)
2. 安装 OpenBLAS via conda
3. 安装 Python 依赖包
4. 配置并构建项目

### 方法 2：手动安装

#### 1. 安装编译器

**选项 A: MinGW（推荐）**

```powershell
# 通过MSYS2安装
# 下载：https://www.msys2.org/
pacman -S mingw-w64-x86_64-gcc
pacman -S mingw-w64-x86_64-cmake
pacman -S mingw-w64-x86_64-make
```

**选项 B: Visual Studio**

```powershell
# 安装Visual Studio Community 2019+
# 选择"C++的桌面开发"工作负载
# 包含MSVC编译器和CMake工具
```

#### 2. 安装依赖

```powershell
# 创建Conda环境
conda create -n cnn_demo python=3.8 -y
conda activate cnn_demo

# 安装OpenBLAS
conda install -c conda-forge openblas -y

# 安装Python依赖
pip install -r requirements.txt

# 验证pybind11
python -c "import pybind11; print('pybind11 OK')"
```

#### 3. 构建项目

```powershell
mkdir build && cd build

# MinGW方式
cmake .. -G "MinGW Makefiles" -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON -DBUILD_PYTHON=ON
cmake --build . -j4

# Visual Studio方式
cmake .. -G "Visual Studio 16 2019" -A x64 -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON
cmake --build . --config Release -j4
```

## Linux 平台

### Ubuntu/Debian

```bash
# 1. 更新软件包
sudo apt-get update

# 2. 安装基础开发工具
sudo apt-get install -y build-essential cmake git

# 3. 安装OpenBLAS
sudo apt-get install -y libopenblas-dev liblapack-dev

# 4. 安装OpenMP (通常已包含在gcc中)
sudo apt-get install -y libomp-dev

# 5. 安装Python开发包 (可选)
sudo apt-get install -y python3-dev python3-pip
pip3 install pybind11 numpy matplotlib

# 6. 构建项目
git clone <repository-url>
cd CNN_demo
mkdir build && cd build
cmake .. -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON -DBUILD_PYTHON=ON
make -j$(nproc)
```

### CentOS/RHEL

```bash
# 1. 安装开发工具
sudo yum groupinstall -y "Development Tools"
sudo yum install -y cmake3

# 2. 安装OpenBLAS
sudo yum install -y openblas-devel lapack-devel

# 3. 安装Python开发包 (可选)
sudo yum install -y python3-devel
pip3 install --user pybind11 numpy matplotlib

# 4. 构建项目
mkdir build && cd build
cmake3 .. -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON -DBUILD_PYTHON=ON
make -j$(nproc)
```

### Arch Linux

```bash
# 安装依赖
sudo pacman -S base-devel cmake openblas lapack python python-pip
pip install pybind11 numpy matplotlib

# 构建项目
mkdir build && cd build
cmake .. -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON -DBUILD_PYTHON=ON
make -j$(nproc)
```

## macOS 平台

### 使用 Homebrew（推荐）

```bash
# 1. 安装Homebrew (如果未安装)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. 安装依赖
brew install cmake gcc openblas python

# 3. 安装Python包
pip3 install pybind11 numpy matplotlib

# 4. 构建项目
mkdir build && cd build
cmake .. -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON -DBUILD_PYTHON=ON
make -j$(sysctl -n hw.ncpu)
```

### 使用 MacPorts

```bash
# 安装依赖
sudo port install cmake gcc12 openblas python39
sudo port select --set python3 python39

# 安装Python包
pip3 install pybind11 numpy matplotlib

# 构建项目（指定编译器）
mkdir build && cd build
cmake .. -DCMAKE_C_COMPILER=gcc-mp-12 -DCMAKE_CXX_COMPILER=g++-mp-12 \
         -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON -DBUILD_PYTHON=ON
make -j$(sysctl -n hw.ncpu)
```

## 依赖检查脚本

在尝试编译之前，运行以下检查脚本确认所有依赖：

### check_dependencies.py

创建此 Python 脚本来检查依赖：

```python
#!/usr/bin/env python3
"""
CNN框架依赖检查脚本
运行前请确保Python 3.7+已安装
"""

import sys
import subprocess
import platform
import os
from pathlib import Path

def check_command(cmd, version_flag="--version"):
    """检查命令是否可用"""
    try:
        result = subprocess.run([cmd, version_flag],
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""

def check_python_package(package):
    """检查Python包是否已安装"""
    try:
        __import__(package)
        return True, ""
    except ImportError as e:
        return False, str(e)

def check_cmake_version(version_str):
    """检查CMake版本是否满足要求"""
    try:
        # 提取版本号 (如 "cmake version 3.20.0")
        version_line = version_str.split('\n')[0]
        version = version_line.split()[-1]
        major, minor = map(int, version.split('.')[:2])
        return major > 3 or (major == 3 and minor >= 15)
    except:
        return False

def main():
    print("🔍 CNN混合架构框架依赖检查")
    print("=" * 50)

    # 系统信息
    print(f"操作系统: {platform.system()} {platform.release()}")
    print(f"架构: {platform.machine()}")
    print(f"Python版本: {sys.version}")
    print()

    all_good = True

    # 必需依赖检查
    print("📋 必需依赖检查")
    print("-" * 30)

    # CMake
    cmake_ok, cmake_info = check_command("cmake")
    if cmake_ok:
        version_ok = check_cmake_version(cmake_info)
        if version_ok:
            print("✅ CMake: 可用且版本满足要求")
        else:
            print("❌ CMake: 版本过低 (需要3.15+)")
            all_good = False
    else:
        print("❌ CMake: 未安装")
        all_good = False

    # C++编译器
    cpp_compilers = ["g++", "clang++", "cl"]
    cpp_found = False
    for compiler in cpp_compilers:
        found, info = check_command(compiler)
        if found:
            print(f"✅ C++编译器: {compiler} 可用")
            cpp_found = True
            break

    if not cpp_found:
        print("❌ C++编译器: 未找到")
        all_good = False

    # C编译器
    c_compilers = ["gcc", "clang", "cl"]
    c_found = False
    for compiler in c_compilers:
        found, info = check_command(compiler)
        if found:
            print(f"✅ C编译器: {compiler} 可用")
            c_found = True
            break

    if not c_found:
        print("❌ C编译器: 未找到")
        all_good = False

    print()

    # 可选依赖检查
    print("🔧 可选依赖检查 (影响性能)")
    print("-" * 30)

    # OpenBLAS检查
    if platform.system() == "Windows":
        # Windows上检查conda环境
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            openblas_path = Path(conda_prefix) / "Library" / "lib" / "openblas.lib"
            if openblas_path.exists():
                print("✅ OpenBLAS: 在Conda环境中找到")
            else:
                print("⚠️  OpenBLAS: 未在Conda环境中找到")
        else:
            print("⚠️  OpenBLAS: 无法检查 (请安装Anaconda)")
    else:
        # Linux/macOS上检查pkg-config
        found, _ = check_command("pkg-config", "--exists openblas")
        if found:
            print("✅ OpenBLAS: 系统已安装")
        else:
            print("⚠️  OpenBLAS: 未找到")

    # OpenMP检查 (通过编译简单程序)
    try:
        test_code = '''
        #include <omp.h>
        #include <iostream>
        int main() {
            #pragma omp parallel
            { std::cout << "OpenMP thread " << omp_get_thread_num() << std::endl; }
            return 0;
        }
        '''

        with open("test_openmp.cpp", "w") as f:
            f.write(test_code)

        # 尝试编译
        result = subprocess.run(["g++", "-fopenmp", "test_openmp.cpp", "-o", "test_openmp"],
                              capture_output=True, timeout=10)

        if result.returncode == 0:
            print("✅ OpenMP: 支持")
        else:
            print("⚠️  OpenMP: 不支持")

        # 清理临时文件
        for f in ["test_openmp.cpp", "test_openmp", "test_openmp.exe"]:
            if Path(f).exists():
                os.remove(f)

    except Exception:
        print("⚠️  OpenMP: 无法检查")

    # Python依赖检查
    print()
    print("🐍 Python依赖检查")
    print("-" * 30)

    python_packages = ["pybind11", "numpy", "matplotlib"]
    python_all_good = True

    for package in python_packages:
        found, error = check_python_package(package)
        if found:
            print(f"✅ {package}: 已安装")
        else:
            print(f"⚠️  {package}: 未安装")
            python_all_good = False

    # 总结
    print()
    print("📊 检查总结")
    print("=" * 50)

    if all_good:
        print("🎉 所有必需依赖都已满足!")
        if python_all_good:
            print("🎯 所有可选依赖也已满足，可以使用完整功能!")
        else:
            print("ℹ️  部分Python依赖缺失，Python绑定将不可用")
    else:
        print("❌ 存在缺失的必需依赖，请先安装后再尝试编译")
        return 1

    print("\n📝 下一步操作:")
    if all_good:
        print("1. 运行构建命令:")
        print("   mkdir build && cd build")
        print("   cmake .. -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON")
        print("   cmake --build . -j4")
    else:
        print("1. 安装缺失的依赖")
        print("2. 重新运行此检查脚本")
        print("3. 确认无误后开始构建")

    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())
```

### 使用检查脚本

```bash
# 下载或创建检查脚本
python3 check_dependencies.py

# 或者直接运行内置的快速检查
python3 -c "
import subprocess
import sys

checks = [
    ('cmake --version', 'CMake'),
    ('g++ --version', 'G++'),
    ('python3 -c \"import pybind11\"', 'pybind11'),
]

for cmd, name in checks:
    try:
        subprocess.check_output(cmd.split(), stderr=subprocess.STDOUT)
        print(f'✅ {name}: OK')
    except:
        print(f'❌ {name}: 缺失')
"
```

## 常见问题与解决方案

### 1. CMake 版本过低

**症状**: `CMake 3.15 or higher is required`

**解决方案**:

```bash
# Ubuntu/Debian
sudo apt-get install cmake

# 或从源码安装最新版本
wget https://github.com/Kitware/CMake/releases/download/v3.25.1/cmake-3.25.1.tar.gz
tar -xzf cmake-3.25.1.tar.gz
cd cmake-3.25.1
./bootstrap && make && sudo make install

# Windows: 从官网下载安装包
# https://cmake.org/download/
```

### 2. OpenBLAS 未找到

**症状**: `Could not find BLAS` 或链接错误

**解决方案**:

```bash
# Ubuntu/Debian
sudo apt-get install libopenblas-dev

# CentOS/RHEL
sudo yum install openblas-devel

# macOS
brew install openblas

# Windows (Anaconda)
conda install -c conda-forge openblas

# 如果还是找不到，可以手动指定路径
cmake .. -DBLAS_LIBRARIES=/path/to/libopenblas.so
```

### 3. OpenMP 不可用

**症状**: 编译时 OpenMP 相关错误

**解决方案**:

```bash
# 安装支持OpenMP的编译器
# Ubuntu/Debian
sudo apt-get install libomp-dev

# CentOS/RHEL
sudo yum install libgomp

# macOS (使用homebrew gcc)
brew install gcc
export CC=gcc-12
export CXX=g++-12

# 或者禁用OpenMP编译
cmake .. -DUSE_OPENMP=OFF
```

### 4. pybind11 找不到

**症状**: `pybind11 not found` 或 Python 绑定构建失败

**解决方案**:

```bash
# 方法1: pip安装
pip install pybind11

# 方法2: conda安装
conda install -c conda-forge pybind11

# 方法3: 从源码安装
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build && cd build
cmake .. -DPYBIND11_TEST=OFF
make -j4 && sudo make install

# 方法4: 禁用Python绑定
cmake .. -DBUILD_PYTHON=OFF
```

### 5. Windows 上 MinGW 问题

**症状**: `'mingw32-make' is not recognized` 或链接错误

**解决方案**:

```powershell
# 确保MinGW在PATH中
$env:PATH += ";C:\msys64\mingw64\bin"

# 或使用完整路径
C:\msys64\mingw64\bin\cmake.exe .. -G "MinGW Makefiles"

# 或者切换到Visual Studio
cmake .. -G "Visual Studio 16 2019" -A x64
```

### 6. 内存不足问题

**症状**: 编译时出现内存不足或系统卡死

**解决方案**:

```bash
# 减少并行编译线程数
cmake --build . -j2  # 而不是 -j4

# 或者单线程编译
cmake --build .

# 增加交换空间 (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

## 性能影响分析

| 配置                      | 编译时间 | 矩阵运算性能 | 内存使用 | 推荐场景   |
| ------------------------- | -------- | ------------ | -------- | ---------- |
| **最小配置** (无可选依赖) | 最快     | 基准 (1x)    | 最低     | 快速测试   |
| **+ OpenMP**              | +20%     | 2-4x         | +10%     | 多核 CPU   |
| **+ OpenBLAS**            | +30%     | 10-50x       | +20%     | 大规模计算 |
| **完整配置**              | +50%     | 50-200x      | +30%     | 生产环境   |

## 后续支持

### CUDA 支持（预览）

当前 CUDA 支持处于开发阶段，可以通过以下方式启用：

```bash
# 安装CUDA Toolkit (10.0+)
# 从NVIDIA官网下载：https://developer.nvidia.com/cuda-downloads

# 启用CUDA编译
cmake .. -DUSE_CUDA=ON -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON

# 注意：当前CUDA实现不完整，主要用于验证
```

### 容器化部署

提供 Docker 支持，简化依赖管理：

```dockerfile
# Dockerfile (预览)
FROM ubuntu:20.04

RUN apt-get update && apt-get install -y \
    build-essential cmake libopenblas-dev \
    python3-dev python3-pip && \
    pip3 install pybind11 numpy matplotlib

COPY . /app
WORKDIR /app
RUN mkdir build && cd build && \
    cmake .. -DUSE_OPENMP=ON -DUSE_OPENBLAS=ON && \
    make -j4

CMD ["./build/bin/simple_cnn"]
```

### 包管理器支持（计划中）

未来将支持更多包管理器：

- **vcpkg** (Windows)
- **Conan** (跨平台)
- **Spack** (HPC 环境)

## 联系支持

如果在依赖安装过程中遇到问题：

1. **检查 FAQ**: 本文档的常见问题部分
2. **运行检查脚本**: 获取详细的环境信息
3. **搜索 Issues**: [GitHub Issues](https://github.com/your-repo/CNN_demo/issues)
4. **创建新 Issue**: 包含检查脚本的输出结果
5. **邮件支持**: your-email@example.com

---

**🔧 记住：依赖问题是可以解决的，不要被编译错误吓到！**
