cmake_minimum_required(VERSION 3.16)
set_property(GLOBAL PROPERTY CXX_STANDARD 20)

##### GENERAL #####
project(global) # required to enable findpackage 
find_package(CUDAToolkit REQUIRED)
find_package(CUDA REQUIRED)
find_package(spdlog CONFIG REQUIRED)
include_directories("/opt/cuda/include/")
include_directories("/opt/cuda/lib64/")
##### GENERAL #####

##### PROJECT 1 #####
project(cuda-1 VERSION 0.0.1)
file(GLOB SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/test1/src/*.cpp
)
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# libraries
target_link_libraries(${PROJECT_NAME} 
    CUDA::cudart
    "cuda"
    "cudnn"
    spdlog::spdlog_header_only    
)
##### PROJECT 1 #####


##### PROJECT 2 #####
project(cuda-2 VERSION 0.0.1)
file(GLOB SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/test2/src/*.cpp
    ${PROJECT_SOURCE_DIR}/test2/src/*.cu

)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75)
cuda_add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# libraries
target_link_libraries(${PROJECT_NAME} 
    CUDA::cudart
    "cuda"
    "cudnn"
    spdlog::spdlog_header_only    
)
##### PROJECT 2 #####

##### PROJECT 3 #####
project(cuda-3 VERSION 0.0.1)
file(GLOB SOURCE_FILES
    ${PROJECT_SOURCE_DIR}/test3/src/*.cpp
    ${PROJECT_SOURCE_DIR}/test3/src/*.cu

)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75)
cuda_add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# add_executable(${PROJECT_NAME} ${SOURCE_FILES})
# libraries
include_directories("${PROJECT_SOURCE_DIR}/test3/vendor/csv-parser")
target_link_libraries(${PROJECT_NAME} 
    CUDA::cudart
    "cuda"
    "cudnn"
    spdlog::spdlog_header_only
)
##### PROJECT 3 #####


# set warning level 4 and warnings treated as errors
if (MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -pedantic -Werror)
endif()