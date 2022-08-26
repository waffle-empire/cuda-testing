#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cuda.h>
#include <cudnn.h>

namespace Wrapper {

    void initGPUData(float *data, int numElements, float value);
    
}