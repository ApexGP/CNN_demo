#!/usr/bin/env python3
"""
CNN混合架构框架依赖检查脚本

检查构建所需的依赖项，包括必需依赖和可选依赖。

使用方法:
    python check_dependencies.py
"""

import sys
import subprocess
import platform
import os
from pathlib import Path
from typing import Tuple, List


def check_command(cmd: str, version_flag: str = "--version") -> Tuple[bool, str]:
    """检查命令是否可用"""
    try:
        result = subprocess.run([cmd, version_flag], 
                              capture_output=True, text=True, timeout=10)
        return result.returncode == 0, result.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, ""


def check_python_package(package: str) -> Tuple[bool, str]:
    """检查Python包是否已安装"""
    try:
        __import__(package)
        module = sys.modules[package]
        version = getattr(module, '__version__', '未知版本')
        return True, version
    except ImportError:
        return False, "未安装"


def check_cmake_version(version_str: str) -> bool:
    """检查CMake版本是否满足要求"""
    try:
        lines = version_str.split('\n')
        for line in lines:
            if 'cmake version' in line.lower():
                version = line.split()[-1]
                major, minor = map(int, version.split('.')[:2])
                return major > 3 or (major == 3 and minor >= 15)
        return False
    except:
        return False


def check_vcpkg() -> Tuple[bool, str]:
    """检查vcpkg是否可用"""
    # 检查vcpkg命令
    found, info = check_command("vcpkg", "version")
    if found:
        return True, f"vcpkg可用: {info.split()[0] if info else '未知版本'}"
    
    # 检查常见的vcpkg安装路径
    possible_paths = [
        "C:/vcpkg/vcpkg.exe",
        "C:/tools/vcpkg/vcpkg.exe",
        "C:/dev/vcpkg/vcpkg.exe",
        os.path.expanduser("~/vcpkg/vcpkg.exe")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return True, f"找到vcpkg: {path}"
    
    return False, "vcpkg未安装或不在PATH中"


def check_vcpkg_package(package_name: str) -> Tuple[bool, str]:
    """检查vcpkg包是否已安装"""
    try:
        result = subprocess.run(
            ["vcpkg", "list", package_name],
            capture_output=True, text=True, timeout=30
        )
        
        if result.returncode == 0 and package_name in result.stdout:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if package_name in line and '[' in line:
                    return True, f"已安装: {line.strip()}"
            return True, "已安装"
        else:
            return False, "未安装"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False, "vcpkg不可用"


def check_openblas() -> Tuple[bool, str]:
    """检查OpenBLAS是否可用"""
    system = platform.system()
    
    # 方法1: 检查vcpkg安装的OpenBLAS
    vcpkg_ok, _ = check_vcpkg()
    if vcpkg_ok:
        openblas_vcpkg, openblas_info = check_vcpkg_package("openblas")
        if openblas_vcpkg:
            return True, f"通过vcpkg安装: {openblas_info}"
    
    if system == "Windows":
        # Windows上检查conda环境
        conda_prefix = os.environ.get("CONDA_PREFIX")
        if conda_prefix:
            possible_paths = [
                Path(conda_prefix) / "Library" / "lib" / "openblas.lib",
                Path(conda_prefix) / "Library" / "lib" / "blas.lib",
                Path(conda_prefix) / "lib" / "libopenblas.dll.a",
            ]
            for path in possible_paths:
                if path.exists():
                    return True, f"在Conda环境中找到: {path}"
            return False, "未在Conda环境中找到OpenBLAS"
        else:
            return False, "未激活Conda环境且vcpkg中未找到"
    else:
        # Linux/macOS上检查
        found, _ = check_command("pkg-config", "--exists openblas")
        if found:
            return True, "通过pkg-config找到"
        
        common_paths = [
            "/usr/lib/libopenblas.so",
            "/usr/lib/x86_64-linux-gnu/libopenblas.so",
            "/usr/local/lib/libopenblas.so",
            "/opt/homebrew/lib/libopenblas.dylib",
            "/usr/local/opt/openblas/lib/libopenblas.dylib",
        ]
        
        for path in common_paths:
            if Path(path).exists():
                return True, f"系统路径找到: {path}"
        
        return False, "未在系统路径中找到"


def check_openmp() -> Tuple[bool, str]:
    """检查OpenMP是否支持"""
    try:
        test_code = '''
#include <omp.h>
#include <iostream>
int main() {
    int num_threads = 0;
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }
    std::cout << "OpenMP threads: " << num_threads << std::endl;
    return 0;
}
'''
        
        temp_file = Path("test_openmp_temp.cpp")
        with open(temp_file, "w") as f:
            f.write(test_code)
        
        compilers = ["g++", "clang++"]
        for compiler in compilers:
            try:
                result = subprocess.run([
                    compiler, "-fopenmp", str(temp_file), 
                    "-o", "test_openmp_temp"
                ], capture_output=True, timeout=30)
                
                if result.returncode == 0:
                    cleanup_temp_files()
                    return True, f"使用{compiler}编译成功"
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue
        
        cleanup_temp_files()
        return False, "编译失败"
        
    except Exception as e:
        cleanup_temp_files()
        return False, f"检查过程出错: {str(e)}"


def check_gtest() -> Tuple[bool, str]:
    """检查Google Test是否可用"""
    # 检查vcpkg安装的gtest
    vcpkg_ok, _ = check_vcpkg()
    if vcpkg_ok:
        gtest_vcpkg, gtest_info = check_vcpkg_package("gtest")
        if gtest_vcpkg:
            return True, f"通过vcpkg安装: {gtest_info}"
    
    # 通过pkgconfig检查
    found, info = check_command("pkg-config", "--modversion gtest")
    if found and "command not found" not in info.lower():
        return True, f"通过pkg-config找到版本 {info.strip()}"
    
    # 未找到
    return False, "未找到，请通过vcpkg安装"


def cleanup_temp_files():
    """清理临时文件"""
    temp_files = [
        "test_openmp_temp.cpp", 
        "test_openmp_temp", 
        "test_openmp_temp.exe"
    ]
    for f in temp_files:
        try:
            if Path(f).exists():
                os.remove(f)
        except:
            pass


def suggest_fixes(issues: List[str]) -> List[str]:
    """为发现的问题提供修复建议"""
    fixes = []
    system = platform.system()
    
    for issue in issues:
        if "cmake" in issue.lower():
            if system == "Windows":
                fixes.append("安装CMake: 下载 https://cmake.org/download/")
            elif system == "Darwin":
                fixes.append("安装CMake: brew install cmake")
            else:
                fixes.append("安装CMake: sudo apt-get install cmake")
        
        elif "编译器" in issue:
            if system == "Windows":
                fixes.append("安装编译器: 安装Visual Studio或MSYS2")
            elif system == "Darwin":
                fixes.append("安装编译器: xcode-select --install")
            else:
                fixes.append("安装编译器: sudo apt-get install build-essential")
        
        elif "openblas" in issue.lower():
            if system == "Windows":
                fixes.append("安装OpenBLAS (推荐): vcpkg install openblas")
                fixes.append("或通过Conda: conda install -c conda-forge openblas")
            else:
                fixes.append("安装OpenBLAS: vcpkg install openblas")
                fixes.append("或系统包管理器安装")
        
        elif "openmp" in issue.lower():
            fixes.append("OpenMP: 确保使用支持OpenMP的编译器")
        
        elif "google test" in issue.lower() or "gtest" in issue.lower():
            fixes.append("安装Google Test (推荐): vcpkg install gtest")
        
        elif "pybind11" in issue.lower():
            fixes.append("安装pybind11: pip install pybind11")
    
    return fixes


def main():
    print("🔍 CNN混合架构框架依赖检查")
    print("=" * 60)
    
    # 系统信息
    print(f"🖥️  操作系统: {platform.system()} {platform.release()}")
    print(f"🏗️  架构: {platform.machine()}")
    print(f"🐍 Python版本: {sys.version.split()[0]}")
    print()
    
    all_good = True
    issues = []
    
    # 必需依赖检查
    print("📋 必需依赖检查")
    print("-" * 40)
    
    # CMake
    cmake_ok, cmake_info = check_command("cmake")
    if cmake_ok:
        version_ok = check_cmake_version(cmake_info)
        if version_ok:
            print("✅ CMake: 可用且版本满足要求")
            for line in cmake_info.split('\n'):
                if 'cmake version' in line.lower():
                    print(f"   版本: {line.split()[-1]}")
                    break
        else:
            print("❌ CMake: 版本过低 (需要3.15+)")
            issues.append("cmake版本过低")
            all_good = False
    else:
        print("❌ CMake: 未安装")
        issues.append("cmake未安装")
        all_good = False
    
    # C++编译器
    cpp_compilers = ["g++", "clang++", "cl"]
    cpp_found = False
    for compiler in cpp_compilers:
        found, info = check_command(compiler)
        if found:
            print(f"✅ C++编译器: {compiler} 可用")
            version_line = info.split('\n')[0] if info else ""
            if version_line:
                print(f"   版本: {version_line}")
            cpp_found = True
            break
    
    if not cpp_found:
        print("❌ C++编译器: 未找到")
        issues.append("C++编译器未找到")
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
        issues.append("C编译器未找到")
        all_good = False
    
    print()
    
    # 可选依赖检查
    print("🔧 可选依赖检查 (影响性能)")
    print("-" * 40)
    
    # vcpkg检查
    vcpkg_ok, vcpkg_info = check_vcpkg()
    if vcpkg_ok:
        print(f"✅ vcpkg: {vcpkg_info}")
    else:
        print(f"⚠️  vcpkg: {vcpkg_info}")
        print("   影响: 无法使用vcpkg管理C++依赖")
    
    # OpenBLAS检查
    openblas_ok, openblas_info = check_openblas()
    if openblas_ok:
        print(f"✅ OpenBLAS: {openblas_info}")
    else:
        print(f"⚠️  OpenBLAS: {openblas_info}")
        print("   影响: 矩阵运算性能降低10-50倍")
        issues.append("OpenBLAS未找到")
    
    # OpenMP检查
    openmp_ok, openmp_info = check_openmp()
    if openmp_ok:
        print(f"✅ OpenMP: {openmp_info}")
    else:
        print(f"⚠️  OpenMP: {openmp_info}")
        print("   影响: 无法使用多线程并行加速")
        issues.append("OpenMP不支持")
    
    # Google Test检查
    gtest_ok, gtest_info = check_gtest()
    if gtest_ok:
        print(f"✅ Google Test: {gtest_info}")
    else:
        print(f"⚠️  Google Test: {gtest_info}")
        print("   影响: 无法构建和运行单元测试")
        issues.append("Google Test未找到")

    print()
    
    # Python依赖检查
    print("🐍 Python依赖检查")
    print("-" * 40)
    
    python_packages = [
        ("pybind11", "Python-C++绑定"),
        ("numpy", "数值计算"),
        ("matplotlib", "可视化")
    ]
    
    for package, description in python_packages:
        found, version = check_python_package(package)
        if found:
            print(f"✅ {package}: 已安装 ({description}) - 版本 {version}")
        else:
            print(f"⚠️  {package}: 未安装 ({description})")
            issues.append(f"{package}未安装")
    
    # 总结
    print()
    print("📊 检查总结")
    print("=" * 60)
    
    if all_good:
        print("🎉 所有必需依赖都已满足!")
        if openblas_ok and openmp_ok:
            print("🎯 性能优化依赖完整，可以使用完整功能!")
        else:
            print("ℹ️  部分性能优化依赖缺失，基本功能可用")
    else:
        print("❌ 存在缺失的必需依赖，请先安装后再尝试编译")
    
    # 显示问题和修复建议
    if issues:
        print("\n🔍 发现的问题:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        fixes = suggest_fixes(issues)
        if fixes:
            print("\n💡 修复建议:")
            for i, fix in enumerate(fixes, 1):
                print(f"  {i}. {fix}")
    
    # 下一步操作指导
    print("\n📝 推荐的构建命令:")
    if platform.system() == "Windows":
        print("  Windows: build.bat --clean --with-openblas --run-tests")
    else:
        print("  Linux:   ./build.sh --clean --with-openblas --run-tests")
    
    print(f"\n📋 详细文档: docs/dependency_guide.md")
    
    return 0 if all_good else 1


if __name__ == "__main__":
    sys.exit(main()) 