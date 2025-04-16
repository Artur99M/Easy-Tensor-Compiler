cmake_minimum_required(VERSION 3.19)

project(EasyTensorCompiler)

set(CMAKE_CXX_FLAGS_DEBUG "-Wall -Wextra -Wpedantic -fsanitize=address,undefined -g -DNDEBUG -O0")

set(src ${CMAKE_SOURCE_DIR}/src/main.cpp
        ${CMAKE_SOURCE_DIR}/src/tensor.cpp
        ${CMAKE_SOURCE_DIR}/src/debug.cpp
        ${CMAKE_SOURCE_DIR}/src/nn_node.cpp
        )

set(exec ETC.out)
add_executable(${exec}
    ${src}
    )
target_compile_features(${exec} PUBLIC cxx_std_17)
target_include_directories(${exec} PUBLIC ${CMAKE_SOURCE_DIR}/include)

include(CTest)
enable_testing()

file(GLOB testfiles "${CMAKE_SOURCE_DIR}/tests/*.dat")
foreach(file ${testfiles})
    add_test (NAME ${file}
        COMMAND python3 ${CMAKE_SOURCE_DIR}/tests/test.py ${CMAKE_SOURCE_DIR} ${file})
    set_tests_properties(${file} PROPERTIES DEPENDS ${exec})
endforeach()

add_subdirectory(${CMAKE_SOURCE_DIR}/tests/unit_tests)
