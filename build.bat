@echo off
echo ========================================
echo 构建 CNN Demo (Windows)
echo ========================================

REM 设置默认值
set CLEAN_BUILD=OFF
set BUILD_TYPE=Debug
set USE_OPENBLAS=OFF
set BUILD_PYTHON=OFF
set RUN_TESTS=OFF
set BUILD_TESTS=ON
set SHOW_HELP=OFF

REM 解析命令行参数
:parse_args
if "%1"=="" goto done_parsing
if "%1"=="--clean" (
    set CLEAN_BUILD=ON
    shift
    goto parse_args
)
if "%1"=="--release" (
    set BUILD_TYPE=Release
    shift
    goto parse_args
)
if "%1"=="--with-openblas" (
    set USE_OPENBLAS=ON
    shift
    goto parse_args
)
if "%1"=="--with-python" (
    set BUILD_PYTHON=ON
    shift
    goto parse_args
)
if "%1"=="--run-tests" (
    set RUN_TESTS=ON
    shift
    goto parse_args
)
if "%1"=="--skip-tests" (
    set BUILD_TESTS=OFF
    shift
    goto parse_args
)
if "%1"=="--help" (
    set SHOW_HELP=ON
    shift
    goto parse_args
)
echo 未知选项: %1
shift
goto parse_args

:done_parsing

if "%SHOW_HELP%"=="ON" (
    echo 用法: %0 [选项]
    echo.
    echo 选项:
    echo   --clean         清理构建目录后重新构建
    echo   --release       使用Release模式构建（默认Debug）
    echo   --with-openblas 启用OpenBLAS支持
    echo   --with-python   构建Python绑定
    echo   --run-tests     构建后运行测试
    echo   --skip-tests    跳过测试构建
    echo   --help          显示此帮助信息
    echo.
    exit /b 0
)

REM 显示构建配置
echo ============================================================
echo 构建配置:
echo   构建类型: %BUILD_TYPE%
echo   清理构建: %CLEAN_BUILD%
echo   OpenBLAS: %USE_OPENBLAS%
echo   Python绑定: %BUILD_PYTHON%
echo   构建测试: %BUILD_TESTS%
echo   运行测试: %RUN_TESTS%
echo ============================================================
echo.

REM 运行依赖检查
echo 🔍 检查依赖...
python scripts\check_dependencies.py
if %ERRORLEVEL% neq 0 (
    echo ❌ 依赖检查失败
    exit /b 1
)

REM 自动检测vcpkg路径
set VCPKG_PATH=
if defined VCPKG_ROOT (
    set VCPKG_PATH=%VCPKG_ROOT%
    echo ✅ 使用VCPKG_ROOT: %VCPKG_PATH%
) else (
    REM 尝试常见的vcpkg安装路径
    if exist "C:\vcpkg\vcpkg.exe" set VCPKG_PATH=C:\vcpkg
    if exist "C:\tools\vcpkg\vcpkg.exe" set VCPKG_PATH=C:\tools\vcpkg
    if exist "%USERPROFILE%\vcpkg\vcpkg.exe" set VCPKG_PATH=%USERPROFILE%\vcpkg
    
    if defined VCPKG_PATH (
        echo ✅ 检测到vcpkg: %VCPKG_PATH%
    ) else (
        echo ⚠️  警告: 未找到vcpkg，某些依赖可能无法找到
    )
)

REM 清理构建目录
if "%CLEAN_BUILD%"=="ON" (
    echo 🧹 清理构建目录...
    if exist build rmdir /s /q build
)

REM 创建构建目录
if not exist build mkdir build

REM 构建CMake参数 - 强制使用MinGW Makefiles
set CMAKE_ARGS=-S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=%BUILD_TYPE%

if defined VCPKG_PATH (
    set CMAKE_ARGS=%CMAKE_ARGS% -DCMAKE_TOOLCHAIN_FILE=%VCPKG_PATH%\scripts\buildsystems\vcpkg.cmake
    set CMAKE_ARGS=%CMAKE_ARGS% -DVCPKG_TARGET_TRIPLET=x64-mingw-static
)

if "%USE_OPENBLAS%"=="ON" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DUSE_OPENBLAS=ON
)

if "%BUILD_PYTHON%"=="ON" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DBUILD_PYTHON=ON
)

if "%BUILD_TESTS%"=="OFF" (
    set CMAKE_ARGS=%CMAKE_ARGS% -DBUILD_TESTS=OFF
)

REM 运行CMake配置
echo 🔧 配置项目...
echo cmake %CMAKE_ARGS%
cmake %CMAKE_ARGS%
if %ERRORLEVEL% neq 0 (
    echo ❌ CMake配置失败
    exit /b 1
)

REM 构建项目
echo 🚀 开始构建...
cmake --build build --config %BUILD_TYPE% --parallel
if %ERRORLEVEL% neq 0 (
    echo ❌ 构建失败
    exit /b 1
)

echo ✅ 构建成功

REM 运行测试
if "%RUN_TESTS%"=="ON" (
    if "%BUILD_TESTS%"=="ON" (
        echo.
        echo 🧪 运行测试...
        cd build
        ctest --output-on-failure --parallel
        if %ERRORLEVEL% neq 0 (
            echo ⚠️  部分测试失败，请检查输出
            cd ..
        ) else (
            echo ✅ 所有测试通过
            cd ..
        )
    ) else (
        echo ⚠️  跳过测试运行（测试构建被禁用）
    )
)

echo.
echo 🎉 构建完成！
echo.
echo 📁 输出目录: build\bin\
echo 📁 库文件: build\lib\
echo.
