#pragma once

#include <cuda_runtime.h>
#include <cudnn.h>

class RNN
{
public:
    RNN(int seqLength, int numLayers,int hiddenSize, int miniBatch, float dropout, bool bidirectional, int mode, int algo);
    ~RNN();

    void setup_input_output();
    void setup_tensor_descriptors();
    void setup_dropout_descriptor();
    void setup_rnn_descriptor();
    void setup_parameters();
    void setup_workspace_memory();
    void setup_weights_and_inputs();
    // *********************************************************************************************************
    // At this point all of the setup is done. We now need to pass through the RNN.
    // *********************************************************************************************************
    void train();

    void print_checksums();
    void calculate_FLOPS();
    void free_gpu_mem();


private:
    int m_GPUCount;
    cudnnHandle_t m_Handle;
    float* m_X;
    int m_ElementCount;

    // Model parameters
    float m_dropout;
    int m_seqLength;
    int m_numLayers;
    int m_hiddenSize;
    int m_inputSize;
    int m_miniBatch;
    bool m_bidirectional;
    int m_mode;
    int m_algo_int;

    // GPU memory allocations
    void* m_x;
    void* m_hx = nullptr;
    void* m_cx = nullptr;

    void *m_dx;
    void *m_dhx = nullptr;
    void *m_dcx = nullptr;

    void *m_y;
    void *m_hy = nullptr;
    void *m_cy = nullptr;

    void *m_dy;
    void *m_dhy = nullptr;
    void *m_dcy = nullptr;

    // Tensor descriptors
    cudnnTensorDescriptor_t* m_xDesc;
    cudnnTensorDescriptor_t* m_yDesc;
    cudnnTensorDescriptor_t* m_dxDesc;
    cudnnTensorDescriptor_t* m_dyDesc;
    cudnnTensorDescriptor_t m_hxDesc, m_cxDesc;
    cudnnTensorDescriptor_t m_hyDesc, m_cyDesc;
    cudnnTensorDescriptor_t m_dhxDesc, m_dcxDesc;
    cudnnTensorDescriptor_t m_dhyDesc, m_dcyDesc;

    // dropout descriptor
    unsigned long long seed = 1337ull; // Pick a seed.
    cudnnDropoutDescriptor_t m_dropoutDesc;

    // rnn descriptor
    cudnnRNNAlgo_t m_algo;
    cudnnRNNDescriptor_t m_rnnDesc;
    cudnnRNNMode_t m_RNNMode;
    cudnnPersistentRNNPlan_t m_rnnPlan;

    // parameters
    void* m_w;
    void* m_dw;
    cudnnFilterDescriptor_t m_wDesc, m_dwDesc;
    size_t m_weightsSize;

    // workspace memory allocation
    size_t m_workSize;
    size_t m_reserveSize;
    void* m_workspace;
    void* m_reserveSpace;

    // something for flops
    int m_numMats;

    // timing for FLOPS calculation
    cudaEvent_t m_start, m_stop;
    float m_timeForward, m_timeBackward1, m_timeBackward2;
};