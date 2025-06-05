#!/bin/bash
# CNN Demo 环境变量设置脚本 (Linux/macOS)
# 用法: source scripts/setup_env.sh

echo "================================="
echo "CNN Demo 环境变量设置"
echo "================================="

# 获取脚本所在目录的父目录作为项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "检测到项目根目录: $PROJECT_ROOT"

# 检查目录是否有效
if [ ! -f "$PROJECT_ROOT/CMakeLists.txt" ]; then
    echo "错误: 未找到CMakeLists.txt，请确保从项目根目录运行此脚本"
    return 1 2>/dev/null || exit 1
fi

# 设置环境变量
export CNN_DEMO_ROOT="$PROJECT_ROOT"
echo "已设置环境变量 CNN_DEMO_ROOT=$CNN_DEMO_ROOT"

# 检查是否已经在shell配置文件中
SHELL_CONFIG=""
if [ -n "$ZSH_VERSION" ]; then
    SHELL_CONFIG="$HOME/.zshrc"
elif [ -n "$BASH_VERSION" ]; then
    if [ -f "$HOME/.bashrc" ]; then
        SHELL_CONFIG="$HOME/.bashrc"
    elif [ -f "$HOME/.bash_profile" ]; then
        SHELL_CONFIG="$HOME/.bash_profile"
    fi
fi

# 询问是否永久设置
if [ -n "$SHELL_CONFIG" ]; then
    echo
    echo "检测到shell配置文件: $SHELL_CONFIG"
    read -p "是否要永久设置此环境变量? (y/n): " PERMANENT
    if [ "$PERMANENT" = "y" ] || [ "$PERMANENT" = "Y" ]; then
        # 检查是否已经存在
        if grep -q "CNN_DEMO_ROOT" "$SHELL_CONFIG"; then
            echo "环境变量已存在于配置文件中，跳过添加"
        else
            echo "" >> "$SHELL_CONFIG"
            echo "# CNN Demo 项目根目录" >> "$SHELL_CONFIG"
            echo "export CNN_DEMO_ROOT=\"$PROJECT_ROOT\"" >> "$SHELL_CONFIG"
            echo "已添加到 $SHELL_CONFIG"
            echo "注意: 重新打开终端或运行 'source $SHELL_CONFIG' 使其生效"
        fi
    fi
else
    echo "警告: 无法检测shell类型，请手动添加以下行到你的shell配置文件:"
    echo "export CNN_DEMO_ROOT=\"$PROJECT_ROOT\""
fi

echo
echo "环境变量设置完成！"
echo "现在可以从任何位置运行 CNN Demo 程序了。"
echo
echo "使用方法:"
echo "  从项目根目录: ./build/bin/mnist_training"
echo "  从任何位置:   mnist_training (如果添加到PATH)"
echo 