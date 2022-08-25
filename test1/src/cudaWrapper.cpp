#include "cudaWrapper.hpp"

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
    int n = 1, c = 1, h = 1, w = 10;
    int m_ElementCount = n*c*h*w;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);

    // create the tensor
    cudaMallocManaged(&m_X, m_ElementCount * sizeof(float));
    for(int i=0;i<m_ElementCount;i++) 
        m_X[i] = i * 1.00f;
    std::cout << "Original array: "; 
    for(int i=0;i<m_ElementCount;i++) 
        std::cout << m_X[i] << " ";

    // create activation function descriptor
    float alpha[1] = {1};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    cudnnActivationForward(
        m_Handle,
        sigmoid_activation,
        alpha,
        x_desc,
        m_X,
        beta,
        x_desc,
        m_X
    );

}

CudaWrapper::~CudaWrapper()
{
    cudnnDestroy(m_Handle);
    spdlog::info("Destroyed cuDNN handle.");
    std::cout << "New array: ";
    for(int i=0;i<m_ElementCount;i++) 
        std::cout << m_X[i] << " ";
    std::cout << std::endl;
    cudaFree(m_X);
}