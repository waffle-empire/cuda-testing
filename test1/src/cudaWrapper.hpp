#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

class CudaWrapper
{
public:
    CudaWrapper();
    ~CudaWrapper();


private:
    int m_GPUCount;
    cudnnHandle_t m_Handle;
    float* m_X;
    int m_ElementCount;
};