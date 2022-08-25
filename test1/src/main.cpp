#include <iostream>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <spdlog/spdlog.h>

int main(int argc, char** argv)
{
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    spdlog::info("Found {} GPUs", numGPUs);
    cudaSetDevice(0); // use GPU0
    int device; 
    struct cudaDeviceProp devProp;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&devProp, device);
    spdlog::info("Compute capability: {}.{}", devProp.major, devProp.minor);

    cudnnHandle_t handle_;
    cudnnCreate(&handle_);
    spdlog::info("Created cuDNN handle");

    // create the tensor descriptor
    cudnnDataType_t dtype = CUDNN_DATA_FLOAT;
    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    int n = 1, c = 1, h = 1, w = 10;
    int NUM_ELEMENTS = n*c*h*w;
    cudnnTensorDescriptor_t x_desc;
    cudnnCreateTensorDescriptor(&x_desc);
    cudnnSetTensor4dDescriptor(x_desc, format, dtype, n, c, h, w);

    // create the tensor
    float *x;
    cudaMallocManaged(&x, NUM_ELEMENTS * sizeof(float));
    for(int i=0;i<NUM_ELEMENTS;i++) 
        x[i] = i * 1.00f;
    std::cout << "Original array: "; 
    for(int i=0;i<NUM_ELEMENTS;i++) 
        std::cout << x[i] << " ";

    // create activation function descriptor
    float alpha[1] = {1};
    float beta[1] = {0.0};
    cudnnActivationDescriptor_t sigmoid_activation;
    cudnnActivationMode_t mode = CUDNN_ACTIVATION_SIGMOID;
    cudnnNanPropagation_t prop = CUDNN_NOT_PROPAGATE_NAN;
    cudnnCreateActivationDescriptor(&sigmoid_activation);
    cudnnSetActivationDescriptor(sigmoid_activation, mode, prop, 0.0f);

    cudnnActivationForward(
        handle_,
        sigmoid_activation,
        alpha,
        x_desc,
        x,
        beta,
        x_desc,
        x
    );

    cudnnDestroy(handle_);
    spdlog::info("Destroyed cuDNN handle.");
    std::cout << "New array: ";
    for(int i=0;i<NUM_ELEMENTS;i++) 
        std::cout << x[i] << " ";
    std::cout << std::endl;
    cudaFree(x);
    return 0;
}