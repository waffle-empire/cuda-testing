// #include <cudnn.h>
// #include <cuda.h>
// #include <stdio.h>


// see https://forums.developer.nvidia.com/t/how-to-call-cuda-function-from-c-file/61986/2

#include "cuda_vars.cuh"


namespace Wrapper {

   __global__ void initGPUData_ker(float *data, int numElements, float value) {
      int tid = blockIdx.x * blockDim.x + threadIdx.x;
      if (tid < numElements) {
         data[tid] = value;
      }
   }

   void initGPUData(float *data, int numElements, float value) {
      dim3 gridDim;
      dim3 blockDim;

      blockDim.x = 1024;
      gridDim.x = (numElements + blockDim.x - 1) / blockDim.x;

      initGPUData_ker <<< gridDim, blockDim >>> (data, numElements, value);

   }
}