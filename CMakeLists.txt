cmake_minimum_required(VERSION 3.16)
project(cuda-1 VERSION 0.0.1)

file(GLOB SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/test1/src/*.cpp
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})

# libraries
find_package(CUDAToolkit REQUIRED)
find_package(spdlog CONFIG REQUIRED)

include_directories("/opt/cuda/include/")
include_directories("/opt/cuda/lib64/") # not necessary


target_link_libraries(${PROJECT_NAME} 
    CUDA::cudart
    "cuda"
    "cudnn"
    spdlog::spdlog_header_only    
)

# set c++20
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 20)

# set warning level 4 and warnings treated as errors
if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()