@echo off
REM CNN Demo 环境变量设置脚本 (Windows)
REM 用法: scripts\setup_env.bat

echo =================================
echo CNN Demo 环境变量设置
echo =================================

REM 获取脚本所在目录的父目录作为项目根目录
set SCRIPT_DIR=%~dp0
REM 移除末尾的反斜杠，然后向上一级目录
set PROJECT_ROOT=%SCRIPT_DIR:~0,-1%
for %%i in ("%PROJECT_ROOT%") do set "PROJECT_ROOT=%%~dpi"
set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

echo 脚本目录: %SCRIPT_DIR:~0,-1%
echo 检测到项目根目录: %PROJECT_ROOT%

REM 检查目录是否有效
if not exist "%PROJECT_ROOT%\CMakeLists.txt" (
    echo 错误: 未找到CMakeLists.txt，请确保脚本结构正确
    echo 当前检测路径: %PROJECT_ROOT%
    echo 请检查项目结构是否为: 项目根目录/scripts/setup_env.bat
    pause
    exit /b 1
)

REM 设置临时环境变量（当前会话有效）
set CNN_DEMO_ROOT=%PROJECT_ROOT%
echo 已设置临时环境变量 CNN_DEMO_ROOT=%CNN_DEMO_ROOT%

REM 询问是否永久设置
echo.
set /p PERMANENT="是否要永久设置此环境变量? (y/n): "
if /i "%PERMANENT%"=="y" (
    setx CNN_DEMO_ROOT "%PROJECT_ROOT%" >nul 2>&1
    if errorlevel 1 (
        echo 警告: 无法设置永久环境变量，可能需要管理员权限
        echo 请手动设置系统环境变量 CNN_DEMO_ROOT=%PROJECT_ROOT%
    ) else (
        echo 已永久设置环境变量 CNN_DEMO_ROOT=%PROJECT_ROOT%
        echo 注意: 新的命令行窗口将生效
    )
)

echo.
echo 环境变量设置完成！
echo 现在可以从任何位置运行 CNN Demo 程序了。
echo.
echo 使用方法:
echo   从项目根目录: .\build\bin\mnist_training.exe
echo   从任何位置:   mnist_training.exe (如果添加到PATH)
echo.
pause 