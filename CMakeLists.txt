cmake_minimum_required(VERSION 3.10)

# project(example)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g")
set(CUDA_DIR /usr/local/cuda-10.2)
set(TRT_ROOT /home/user/Software/TensorRT-7.1.3.4)

# find_package(CUDA REQUIRED)  # cuda compiler needed

include(CheckLanguage)  # cuda语法需要
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
else ()
    message(ERROR "CUDA Compiler not find.")
endif()

#----------- Set architecture and CUDA -----------#
# 可以根据自己的机器只写一个
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS}
        -O3
        -gencode arch=compute_30,code=sm_30
        -gencode arch=compute_35,code=sm_35
        -gencode arch=compute_50,code=sm_50
        -gencode arch=compute_52,code=sm_52
        -gencode arch=compute_61,code=sm_61
        -gencode arch=compute_62,code=sm_62
        -gencode arch=compute_70,code=sm_70
        -gencode arch=compute_72,code=sm_72
        )

find_library(CUDART cudart HINTS ${CUDA_DIR}/lib64)

include_directories(/usr/local/include
                    ${PROJECT_SOURCE_DIR}/cuda_sample
                    ${PROJECT_SOURCE_DIR}/rt_api
                    ${CUDA_INCLUDE_DIRS}
                    ${TRT_ROOT}/include
                    ${CUDA_DIR}/include)

#----------------------------------------------#

file(GLOB SRC ${PROJECT_SOURCE_DIR}/rt_api/*.cpp)
file(GLOB CU_SRC ${PROJECT_SOURCE_DIR}/cuda_sample/*.cu)

link_directories(/usr/local/lib ${CUDA_DIR}/lib64 ${TRT_ROOT}/lib)

add_executable(example1 example1.cpp ${CU_SRC})
add_executable(example2 example2.cpp ${CU_SRC})
# cuda_add_executable(test2 test2.cpp ${CU_SRC})  # 这也ok,需要find CUDA依赖
add_executable(example3 example3.cpp ${SRC})
target_link_libraries(example3 nvinfer ${CUDART})