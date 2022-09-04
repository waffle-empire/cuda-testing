#pragma once

#include "structs/measurement.hpp"

#include <csv.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

class RNN
{
public:
	RNN(std::filesystem::path file_path, int seqLength, int numLayers, int hiddenSize, int miniBatch, float dropout, bool bidirectional, int mode, int algo);
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

	bool read_input_file();
	const char* file_name();

    void standard_scale(std::vector<std::unique_ptr<measurement>>*);
    inline void clone_measurement(measurement* dst, measurement* src);
    std::vector<std::vector<std::unique_ptr<measurement>>> generate_sequences();


private:
	std::filesystem::path m_input_file;

	cudnnHandle_t m_Handle;
	int m_GPUCount;
	int m_ElementCount;
	float* m_X;

	// Model parameters
	int m_seqLength;
	int m_numLayers;
	int m_hiddenSize;
	int m_inputSize;
	int m_miniBatch;
	int m_mode;
	int m_algo_int;
	float m_dropout;
	bool m_bidirectional;

	// GPU memory allocations
	void* m_x;
	void* m_hx = nullptr;
	void* m_cx = nullptr;

	void* m_dx;
	void* m_dhx = nullptr;
	void* m_dcx = nullptr;

	void* m_y;
	void* m_hy = nullptr;
	void* m_cy = nullptr;

	void* m_dy;
	void* m_dhy = nullptr;
	void* m_dcy = nullptr;

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
	unsigned long long seed = 1337ull;// Pick a seed.
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

    // csv data related
    std::unique_ptr<measurement> m_measurements_sums; // all summed values of the data, used for scaling
    std::unique_ptr<measurement> m_measurements_std_devs; // all std dev values of the data, used for scaling
    std::unique_ptr<measurement> m_measurements_means; // all mean values of the data, used for scaling

	std::vector<std::unique_ptr<measurement>> m_measurements_All;
    // X are features, Y are labels
	std::vector<std::unique_ptr<measurement>> m_measurements_X_train;
	std::vector<std::unique_ptr<int>> m_targets_Y_train;

	std::vector<std::unique_ptr<measurement>> m_measurements_X_test;
    std::vector<std::unique_ptr<int>> m_targets_Y_test;

    std::vector<std::unique_ptr<measurement>> m_measurements_X_train_scaled;
	std::vector<std::unique_ptr<measurement>> m_measurements_X_test_scaled;


};