#include "cudaWrapper.hpp"

int main(int argc, char** argv)
{
    CudaWrapper* cudaWrapper{new CudaWrapper()};

    delete cudaWrapper;
    cudaWrapper = nullptr;

    return 0;
}