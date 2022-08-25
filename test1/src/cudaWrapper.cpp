#include "cudaWrapper.hpp"

#include <cudnn_backend.h>
#include <cudnn_ops_infer.h>
#include <iostream>
#include <spdlog/spdlog.h>

CudaWrapper::CudaWrapper() :
    m_GPUCount{0},
    m_ElementCount(0)
{
    cudaGetDeviceCount(&m_GPUCount);
    spdlog::info("Found {} GPUs", m_GPUCount);
    cudaSetDevice(0); // use GPU0
    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    spdlog::info("Compute capability: {}.{}", devProp.major, devProp.minor);

    cudnnCreate(&m_Handle);
    spdlog::info("Created cuDNN handle");

    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = 1, c = 5, h = 1, w = 17;
    int m_ElementCount = n*c*h*w;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);

    // create the tensor
    cudaMallocManaged(&m_X, m_ElementCount * sizeof(float));
    for(int i=0;i<m_ElementCount;i++) 
        m_X[i] = i * 0.10f - 3;
    std::cout << "Original array: "; 
    for(int i=0;i<m_ElementCount;i++) 
        std::cout << m_X[i] << " ";
    std::cout << '\n';
    

    // create activation function descriptor
    float alpha[1] = {0};
    float beta[1] = {0};
    cudnnActivationDescriptor_t relu_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_RELU;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&relu_activation);
    cudnnSetActivationDescriptor(relu_activation, mode, prop, 0.0f);

    cudnnStatus_t result = cudnnActivationForward(
        m_Handle,
        relu_activation,
        alpha,
        x_desc,
        m_X,
        beta,
        x_desc,
        m_X
    );

    std::cout << cudnnGetErrorString(result) << '\n';

    std::cout << "New array: ";
    for(int i=0;i<m_ElementCount;i++) 
        std::cout << m_X[i] << " ";
    std::cout << '\n';
    cudaFree(m_X);
}

CudaWrapper::~CudaWrapper()
{
    spdlog::info("Destroyed cuDNN handle.");
    cudnnDestroy(m_Handle);
}