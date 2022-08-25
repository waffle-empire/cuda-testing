cmake_minimum_required(VERSION 3.10)
project(cuda-1 VERSION 0.0.1)

file(GLOB SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/test1/src/*.cpp
)

find_package(CUDAToolkit REQUIRED)

include_directories("/opt/cuda/include/")
include_directories("/opt/cuda/lib64/") # not necessary

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_link_libraries(${PROJECT_NAME} CUDA::cudart)
target_link_libraries(${PROJECT_NAME} "cuda")
target_link_libraries(${PROJECT_NAME} "cudnn")



set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)