# 示例目录的CMakeLists.txt

# 添加C++示例子目录
add_subdirectory(cpp_examples)

# Python示例不需要编译，只需复制到构建目录
if(BUILD_PYTHON AND Python_FOUND)
    file(GLOB PYTHON_EXAMPLES python_examples/*.py)
    foreach(EXAMPLE ${PYTHON_EXAMPLES})
        get_filename_component(EXAMPLE_NAME ${EXAMPLE} NAME)
        configure_file(
            ${EXAMPLE}
            ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${EXAMPLE_NAME}
            COPYONLY
        )
    endforeach()
endif() 