#!/bin/bash
# CNN混合架构框架 - Linux构建脚本
# 
# 使用方法:
#   ./build_linux.sh [选项]
#
# 选项:
#   --clean         清理之前的构建
#   --release       Release构建（默认Debug）
#   --with-openblas 启用OpenBLAS支持
#   --with-python   构建Python绑定
#   --run-tests     构建后运行测试
#   --help          显示此帮助信息

set -e  # 遇到错误立即退出

# 默认设置
BUILD_TYPE="Debug"
USE_OPENBLAS="OFF"
BUILD_PYTHON="OFF"
RUN_TESTS="OFF"
CLEAN_BUILD="OFF"
SHOW_HELP="OFF"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_BUILD="ON"
            shift
            ;;
        --release)
            BUILD_TYPE="Release"
            shift
            ;;
        --with-openblas)
            USE_OPENBLAS="ON"
            shift
            ;;
        --with-python)
            BUILD_PYTHON="ON"
            shift
            ;;
        --run-tests)
            RUN_TESTS="ON"
            shift
            ;;
        --help)
            SHOW_HELP="ON"
            shift
            ;;
        *)
            echo "错误: 未知参数 $1"
            echo "使用 --help 查看可用选项"
            exit 1
            ;;
    esac
done

# 显示帮助信息
if [[ "$SHOW_HELP" == "ON" ]]; then
    echo "CNN混合架构框架 - Linux构建脚本"
    echo ""
    echo "使用方法: ./build_linux.sh [选项]"
    echo ""
    echo "选项:"
    echo "  --clean         清理之前的构建"
    echo "  --release       Release构建（默认Debug）"
    echo "  --with-openblas 启用OpenBLAS支持"
    echo "  --with-python   构建Python绑定"
    echo "  --run-tests     构建后运行测试"
    echo "  --help          显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  ./build_linux.sh --clean --release --with-openblas --run-tests"
    echo "  ./build_linux.sh --with-python"
    exit 0
fi

echo ""
echo "🚀 CNN混合架构框架构建脚本 (Linux)"
echo "============================================================"
echo "构建类型: $BUILD_TYPE"
echo "OpenBLAS: $USE_OPENBLAS"
echo "Python绑定: $BUILD_PYTHON"
echo "运行测试: $RUN_TESTS"
echo "============================================================"
echo ""

# 检查依赖
echo "🔍 检查构建依赖..."
python3 scripts/check_dependencies.py
if [[ $? -ne 0 ]]; then
    echo "❌ 依赖检查失败，请先安装缺失的依赖"
    exit 1
fi

# 清理构建目录
if [[ "$CLEAN_BUILD" == "ON" ]]; then
    echo ""
    echo "🧹 清理构建目录..."
    rm -rf build install
    echo "✅ 清理完成"
fi

# 创建构建目录
echo ""
echo "📁 创建构建目录..."
mkdir -p build
cd build

# 检测vcpkg
VCPKG_CMAKE_ARG=""
if [[ -n "$VCPKG_ROOT" ]]; then
    echo "📦 检测到vcpkg: $VCPKG_ROOT"
    VCPKG_CMAKE_ARG="-DCMAKE_TOOLCHAIN_FILE=$VCPKG_ROOT/scripts/buildsystems/vcpkg.cmake"
elif [[ -f "/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake" ]]; then
    echo "📦 找到vcpkg: /usr/local/vcpkg"
    VCPKG_CMAKE_ARG="-DCMAKE_TOOLCHAIN_FILE=/usr/local/vcpkg/scripts/buildsystems/vcpkg.cmake"
elif [[ -f "/opt/vcpkg/scripts/buildsystems/vcpkg.cmake" ]]; then
    echo "📦 找到vcpkg: /opt/vcpkg"
    VCPKG_CMAKE_ARG="-DCMAKE_TOOLCHAIN_FILE=/opt/vcpkg/scripts/buildsystems/vcpkg.cmake"
elif [[ -f "$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake" ]]; then
    echo "📦 找到vcpkg: $HOME/vcpkg"
    VCPKG_CMAKE_ARG="-DCMAKE_TOOLCHAIN_FILE=$HOME/vcpkg/scripts/buildsystems/vcpkg.cmake"
else
    echo "⚠️  未检测到vcpkg，某些依赖可能需要手动安装"
fi

# 配置项目
echo ""
echo "⚙️ 配置项目..."
cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DUSE_OPENMP=ON \
    -DUSE_OPENBLAS="$USE_OPENBLAS" \
    -DBUILD_PYTHON="$BUILD_PYTHON" \
    -DBUILD_TESTS=ON \
    -DBUILD_EXAMPLES=ON \
    $VCPKG_CMAKE_ARG

if [[ $? -ne 0 ]]; then
    echo "❌ 配置失败"
    cd ..
    exit 1
fi

# 构建项目
echo ""
echo "🔨 构建项目..."
CORES=$(nproc 2>/dev/null || echo 4)
cmake --build . -j$CORES

if [[ $? -ne 0 ]]; then
    echo "❌ 构建失败"
    cd ..
    exit 1
fi

echo "✅ 构建成功！"

# 运行测试
if [[ "$RUN_TESTS" == "ON" ]]; then
    echo ""
    echo "🧪 运行测试..."
    ctest --output-on-failure
    if [[ $? -ne 0 ]]; then
        echo "⚠️ 某些测试失败"
    else
        echo "✅ 所有测试通过"
    fi
fi

# 回到项目根目录
cd ..

echo ""
echo "🎉 构建完成！"
echo ""
echo "📝 生成的文件:"
echo "  • 可执行文件: build/bin/"
echo "  • 库文件: build/lib/"
if [[ "$BUILD_PYTHON" == "ON" ]]; then
    echo "  • Python模块: build/python/"
fi
echo ""
echo "💡 下一步:"
echo "  • 运行示例: ./build/bin/cnn_example"
echo "  • 运行测试: ./build/bin/test_*"
if [[ "$BUILD_PYTHON" == "ON" ]]; then
    echo "  • 使用Python模块: import sys; sys.path.append('build/python'); import cnn"
fi
echo "" 