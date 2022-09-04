#include "rnn.hpp"

#include "cuda_vars.cuh"

#include <cudnn_backend.h>
#include <cudnn_ops_infer.h>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <spdlog/spdlog.h>


// Define some error checking macros.
#define cudaErrCheck(stat)                         \
	{                                              \
		cudaErrCheck_((stat), __FILE__, __LINE__); \
	}
void cudaErrCheck_(cudaError_t stat, const char* file, int line)
{
	if (stat != cudaSuccess)
	{
		fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
	}
}

#define cudnnErrCheck(stat)                         \
	{                                               \
		cudnnErrCheck_((stat), __FILE__, __LINE__); \
	}
void cudnnErrCheck_(cudnnStatus_t stat, const char* file, int line)
{
	if (stat != CUDNN_STATUS_SUCCESS)
	{
		fprintf(stderr, "cuDNN Error: %s %s %d\n", cudnnGetErrorString(stat), file, line);
	}
}


RNN::RNN(std::filesystem::path file_path, int seqLength, int numLayers, int hiddenSize, int miniBatch, float dropout, bool bidirectional, int mode, int algo) :
m_input_file(file_path), m_GPUCount{0}, m_ElementCount(0), m_seqLength(seqLength), m_numLayers(numLayers), m_hiddenSize(hiddenSize), m_inputSize(hiddenSize), m_miniBatch(miniBatch), m_dropout(dropout), m_bidirectional(bidirectional), m_mode(mode), m_algo_int(algo)
{
	cudaGetDeviceCount(&m_GPUCount);
	spdlog::info("Found {} GPUs", m_GPUCount);
	cudaSetDevice(0);// use GPU0

	int device;
	struct cudaDeviceProp devProp;
	cudaGetDevice(&device);
	cudaGetDeviceProperties(&devProp, device);
	spdlog::info("Compute capability: {}.{}", devProp.major, devProp.minor);

	cudnnErrCheck(cudnnCreate(&m_Handle));
	spdlog::info("Created cuDNN handle");
}

void RNN::setup_input_output()
{
	// Memory allocation. hx, cx, dhx, dcx, hy, cy, dhy and dcy can be NULL.
	cudaErrCheck(cudaMalloc((void**)&m_x, m_seqLength * m_inputSize * m_miniBatch * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&m_hx, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&m_cx, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));

	cudaErrCheck(cudaMalloc((void**)&m_dx, m_seqLength * m_inputSize * m_miniBatch * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&m_dhx, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&m_dcx, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));

	cudaErrCheck(cudaMalloc((void**)&m_y, m_seqLength * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&m_hy, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&m_cy, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));

	cudaErrCheck(cudaMalloc((void**)&m_dy, m_seqLength * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&m_dhy, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));
	cudaErrCheck(cudaMalloc((void**)&m_dcy, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1) * sizeof(float)));
}

void RNN::setup_tensor_descriptors()
{
	m_xDesc  = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_yDesc  = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_dxDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));
	m_dyDesc = (cudnnTensorDescriptor_t*)malloc(m_seqLength * sizeof(cudnnTensorDescriptor_t));

	int dimA[3];
	int strideA[3];

	// In this example dimA[1] is constant across the whole sequence
	// This isn't required, all that is required is that it does not increase.
	for (int i = 0; i < m_seqLength; i++)
	{
		cudnnErrCheck(cudnnCreateTensorDescriptor(&m_xDesc[i]));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&m_yDesc[i]));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&m_dxDesc[i]));
		cudnnErrCheck(cudnnCreateTensorDescriptor(&m_dyDesc[i]));

		dimA[0] = m_miniBatch;
		dimA[1] = m_inputSize;
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		cudnnErrCheck(cudnnSetTensorNdDescriptor(m_xDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(m_dxDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));

		dimA[0] = m_miniBatch;
		dimA[1] = m_bidirectional ? m_hiddenSize * 2 : m_hiddenSize;
		dimA[2] = 1;

		strideA[0] = dimA[2] * dimA[1];
		strideA[1] = dimA[2];
		strideA[2] = 1;

		cudnnErrCheck(cudnnSetTensorNdDescriptor(m_yDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
		cudnnErrCheck(cudnnSetTensorNdDescriptor(m_dyDesc[i], CUDNN_DATA_FLOAT, 3, dimA, strideA));
	}

	dimA[0] = m_numLayers * (m_bidirectional ? 2 : 1);
	dimA[1] = m_miniBatch;
	dimA[2] = m_hiddenSize;

	strideA[0] = dimA[2] * dimA[1];
	strideA[1] = dimA[2];
	strideA[2] = 1;

	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_hxDesc));
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_cxDesc));
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_hyDesc));
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_cyDesc));
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_dhxDesc));
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_dcxDesc));
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_dhyDesc));
	cudnnErrCheck(cudnnCreateTensorDescriptor(&m_dcyDesc));

	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_hxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_cxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_hyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_cyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_dhxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_dcxDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_dhyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
	cudnnErrCheck(cudnnSetTensorNdDescriptor(m_dcyDesc, CUDNN_DATA_FLOAT, 3, dimA, strideA));
}

void RNN::setup_dropout_descriptor()
{
	cudnnErrCheck(cudnnCreateDropoutDescriptor(&m_dropoutDesc));

	// How much memory does dropout need for states?
	// These states are used to generate random numbers internally
	// and should not be freed until the RNN descriptor is no longer used
	size_t stateSize;
	void* states;

	cudnnErrCheck(cudnnDropoutGetStatesSize(m_Handle, &stateSize));

	cudaErrCheck(cudaMalloc(&states, stateSize));

	cudnnErrCheck(cudnnSetDropoutDescriptor(m_dropoutDesc, m_Handle, m_dropout, states, stateSize, seed));
}

void RNN::setup_rnn_descriptor()
{
	cudnnErrCheck(cudnnCreateRNNDescriptor(&m_rnnDesc));
	if (m_mode == 0)
		m_RNNMode = CUDNN_RNN_RELU;
	else if (m_mode == 1)
		m_RNNMode = CUDNN_RNN_TANH;
	else if (m_mode == 2)
		m_RNNMode = CUDNN_LSTM;
	else if (m_mode == 3)
		m_RNNMode = CUDNN_GRU;


	if (m_algo_int == 0)
		m_algo = CUDNN_RNN_ALGO_STANDARD;
	else if (m_algo_int == 1)
		m_algo = CUDNN_RNN_ALGO_PERSIST_STATIC;
	else if (m_algo_int == 2)
		m_algo = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;

	//cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_STANDARD;
	//cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_PERSIST_STATIC;
	//cudnnRNNAlgo_t algo = CUDNN_RNN_ALGO_PERSIST_DYNAMIC;
	cudnnErrCheck(cudnnSetRNNDescriptor_v6(m_Handle, m_rnnDesc, m_hiddenSize, m_numLayers, m_dropoutDesc, CUDNN_LINEAR_INPUT,// We can also skip the input matrix transformation
	    m_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL, m_RNNMode, m_algo,//CUDNN_RNN_ALGO_STANDARD,
	    CUDNN_DATA_FLOAT));
	// EDIT HERE
	if (m_algo_int == 2)
	{
		cudnnPersistentRNNPlan_t rnnDescPlan;
		cudnnErrCheck(cudnnCreatePersistentRNNPlan(m_rnnDesc, m_miniBatch, CUDNN_DATA_FLOAT, &rnnDescPlan));
		cudnnErrCheck(cudnnSetPersistentRNNPlan(m_rnnDesc, rnnDescPlan));
		printf("dynamic set\n");
	}
	printf("algo : %d    ", m_algo);
}

void RNN::setup_parameters()
{
	// This needs to be done after the rnn descriptor is set as otherwise
	// we don't know how many parameters we have to allocate
	cudnnErrCheck(cudnnCreateFilterDescriptor(&m_wDesc));
	cudnnErrCheck(cudnnCreateFilterDescriptor(&m_dwDesc));


	cudnnErrCheck(cudnnGetRNNParamsSize(m_Handle, m_rnnDesc, m_xDesc[0], &m_weightsSize, CUDNN_DATA_FLOAT));

	int dimW[3];
	dimW[0] = m_weightsSize / sizeof(float);
	dimW[1] = 1;
	dimW[2] = 1;

	cudnnErrCheck(cudnnSetFilterNdDescriptor(m_wDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
	cudnnErrCheck(cudnnSetFilterNdDescriptor(m_dwDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));

	cudaErrCheck(cudaMalloc((void**)&m_w, m_weightsSize));
	cudaErrCheck(cudaMalloc((void**)&m_dw, m_weightsSize));
}

void RNN::setup_workspace_memory()
{
	// Need for every pass
	cudnnErrCheck(cudnnGetRNNWorkspaceSize(m_Handle, m_rnnDesc, m_seqLength, m_xDesc, &m_workSize));
	// Only needed in training, shouldn't be touched between passes.
	cudnnErrCheck(cudnnGetRNNTrainingReserveSize(m_Handle, m_rnnDesc, m_seqLength, m_xDesc, &m_reserveSize));

	cudaErrCheck(cudaMalloc((void**)&m_workspace, m_workSize));
	cudaErrCheck(cudaMalloc((void**)&m_reserveSpace, m_reserveSize));
}

void RNN::setup_weights_and_inputs()
{
	// *********************************************************************************************************
	// Initialise weights and inputs
	// *********************************************************************************************************
	// We initialise to something simple.
	// Matrices are initialised to 1 / matrixSize, biases to 1, data is 1.
	Wrapper::initGPUData((float*)m_x, m_seqLength * m_inputSize * m_miniBatch, 1.f);
	if (m_hx != NULL)
		Wrapper::initGPUData((float*)m_hx, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1), 1.f);
	if (m_cx != NULL)
		Wrapper::initGPUData((float*)m_cx, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1), 1.f);

	Wrapper::initGPUData((float*)m_dy, m_seqLength * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1), 1.f);
	if (m_dhy != NULL)
		Wrapper::initGPUData((float*)m_dhy, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1), 1.f);
	if (m_dcy != NULL)
		Wrapper::initGPUData((float*)m_dcy, m_numLayers * m_hiddenSize * m_miniBatch * (m_bidirectional ? 2 : 1), 1.f);


	// Weights
	int numLinearLayers = 0;
	if (m_RNNMode == CUDNN_RNN_RELU || m_RNNMode == CUDNN_RNN_TANH)
	{
		numLinearLayers = 2;
	}
	else if (m_RNNMode == CUDNN_LSTM)
	{
		numLinearLayers = 8;
	}
	else if (m_RNNMode == CUDNN_GRU)
	{
		numLinearLayers = 6;
	}

	for (int layer = 0; layer < m_numLayers * (m_bidirectional ? 2 : 1); layer++)
	{
		for (int linLayerID = 0; linLayerID < numLinearLayers; linLayerID++)
		{
			cudnnFilterDescriptor_t linLayerMatDesc;
			cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerMatDesc));
			float* linLayerMat;

			cudnnErrCheck(cudnnGetRNNLinLayerMatrixParams(m_Handle, m_rnnDesc, layer, m_xDesc[0], m_wDesc, m_w, linLayerID, linLayerMatDesc, (void**)&linLayerMat));

			cudnnDataType_t dataType;
			cudnnTensorFormat_t format;
			int nbDims;
			int filterDimA[3];
			cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerMatDesc, 3, &dataType, &format, &nbDims, filterDimA));

			Wrapper::initGPUData(linLayerMat, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f / (float)(filterDimA[0] * filterDimA[1] * filterDimA[2]));

			cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerMatDesc));

			cudnnFilterDescriptor_t linLayerBiasDesc;
			cudnnErrCheck(cudnnCreateFilterDescriptor(&linLayerBiasDesc));
			float* linLayerBias;

			cudnnErrCheck(cudnnGetRNNLinLayerBiasParams(m_Handle, m_rnnDesc, layer, m_xDesc[0], m_wDesc, m_w, linLayerID, linLayerBiasDesc, (void**)&linLayerBias));

			cudnnErrCheck(cudnnGetFilterNdDescriptor(linLayerBiasDesc, 3, &dataType, &format, &nbDims, filterDimA));

			Wrapper::initGPUData(linLayerBias, filterDimA[0] * filterDimA[1] * filterDimA[2], 1.f);

			cudnnErrCheck(cudnnDestroyFilterDescriptor(linLayerBiasDesc));
		}
	}
	// *********************************************************************************************************
	// Dynamic persistent RNN plan (if using this algo)
	// *********************************************************************************************************

	if (m_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC)
	{
		// Note: This step is expensive. Once completed the plan can be reused so long as the descriptor
		//       minibatch or datatype don't change.
		cudnnErrCheck(cudnnCreatePersistentRNNPlan(m_rnnDesc, m_miniBatch, CUDNN_DATA_FLOAT, &m_rnnPlan));
		// Tell calls using this descriptor which plan to use.
		cudnnErrCheck(cudnnSetPersistentRNNPlan(m_rnnDesc, m_rnnPlan));
	}
}

void RNN::train()
{
	cudaErrCheck(cudaDeviceSynchronize());

	cudaErrCheck(cudaEventCreate(&m_start));
	cudaErrCheck(cudaEventCreate(&m_stop));

	cudaErrCheck(cudaEventRecord(m_start));

	// If we're not training we use this instead
	// cudnnErrCheck(cudnnRNNForwardInference(cudnnHandle,
	// rnnDesc,
	// seqLength,
	// xDesc,
	// x,
	// hxDesc,
	// hx,
	// cxDesc,
	// cx,
	// wDesc,
	// w,
	// yDesc,
	// y,
	// hyDesc,
	// hy,
	// cyDesc,
	// cy,
	// workspace,
	// workSize));

	cudnnErrCheck(cudnnRNNForwardTraining(m_Handle, m_rnnDesc, m_seqLength, m_xDesc, m_x, m_hxDesc, m_hx, m_cxDesc, m_cx, m_wDesc, m_w, m_yDesc, m_y, m_hyDesc, m_hy, m_cyDesc, m_cy, m_workspace, m_workSize, m_reserveSpace, m_reserveSize));

	cudaErrCheck(cudaEventRecord(m_stop));
	cudaErrCheck(cudaEventSynchronize(m_stop));
	cudaErrCheck(cudaEventElapsedTime(&m_timeForward, m_start, m_stop));

	cudaErrCheck(cudaEventRecord(m_start));

	cudnnErrCheck(cudnnRNNBackwardData(m_Handle, m_rnnDesc, m_seqLength, m_yDesc, m_y, m_dyDesc, m_dy, m_dhyDesc, m_dhy, m_dcyDesc, m_dcy, m_wDesc, m_w, m_hxDesc, m_hx, m_cxDesc, m_cx, m_dxDesc, m_dx, m_dhxDesc, m_dhx, m_dcxDesc, m_dcx, m_workspace, m_workSize, m_reserveSpace, m_reserveSize));

	cudaErrCheck(cudaEventRecord(m_stop));
	cudaErrCheck(cudaEventSynchronize(m_stop));
	cudaErrCheck(cudaEventElapsedTime(&m_timeBackward1, m_start, m_stop));

	cudaErrCheck(cudaEventRecord(m_start));

	// cudnnRNNBackwardWeights adds to the data in dw.
	cudaErrCheck(cudaMemset(m_dw, 0, m_weightsSize));

	cudnnErrCheck(cudnnRNNBackwardWeights(m_Handle, m_rnnDesc, m_seqLength, m_xDesc, m_x, m_hxDesc, m_hx, m_yDesc, m_y, m_workspace, m_workSize, m_dwDesc, m_dw, m_reserveSpace, m_reserveSize));


	cudaErrCheck(cudaEventRecord(m_stop));

	cudaErrCheck(cudaEventSynchronize(m_stop));
	cudaErrCheck(cudaEventElapsedTime(&m_timeBackward2, m_start, m_stop));
}

void RNN::print_checksums()
{
	// *********************************************************************************************************
	// Print checksums.
	// *********************************************************************************************************
	if (true)
	{
		float* testOutputi;
		float* testOutputh;
		float* testOutputc;

		int biDirScale = (m_bidirectional ? 2 : 1);

		testOutputi = (float*)malloc(m_hiddenSize * m_seqLength * m_miniBatch * biDirScale * sizeof(float));
		testOutputh = (float*)malloc(m_hiddenSize * m_miniBatch * m_numLayers * biDirScale * sizeof(float));
		testOutputc = (float*)malloc(m_hiddenSize * m_miniBatch * m_numLayers * biDirScale * sizeof(float));

		cudaErrCheck(cudaMemcpy(testOutputi, m_y, m_hiddenSize * m_seqLength * m_miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));
		if (m_hy != NULL)
			cudaErrCheck(cudaMemcpy(testOutputh, m_hy, m_numLayers * m_hiddenSize * m_miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));
		if (m_cy != NULL && m_RNNMode == CUDNN_LSTM)
			cudaErrCheck(cudaMemcpy(testOutputc, m_cy, m_numLayers * m_hiddenSize * m_miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));

		double checksumi = 0.f;
		double checksumh = 0.f;
		double checksumc = 0.f;

		for (int m = 0; m < m_miniBatch; m++)
		{
			double localSumi = 0;
			double localSumh = 0;
			double localSumc = 0;

			for (int j = 0; j < m_seqLength; j++)
			{
				for (int i = 0; i < m_hiddenSize * biDirScale; i++)
				{
					localSumi += testOutputi[j * m_miniBatch * m_hiddenSize * biDirScale + m * m_hiddenSize * biDirScale + i];
				}
			}
			for (int j = 0; j < m_numLayers * biDirScale; j++)
			{
				for (int i = 0; i < m_hiddenSize; i++)
				{
					if (m_hy != NULL)
						localSumh += testOutputh[j * m_hiddenSize * m_miniBatch + m * m_hiddenSize + i];
					if (m_cy != NULL)
						if (m_RNNMode == CUDNN_LSTM)
							localSumc += testOutputc[j * m_hiddenSize * m_miniBatch + m * m_hiddenSize + i];
				}
			}

			checksumi += localSumi;
			checksumh += localSumh;
			checksumc += localSumc;
		}

		printf("i checksum %E     ", checksumi);
		// fprintf(fp,"i checksum %E     ", checksumi);
		if (m_RNNMode == CUDNN_LSTM)
		{
			printf("c checksum %E     ", checksumc);
			// fprintf(fp,"c checksum %E     ", checksumc);
		}
		printf("h checksum %E\n", checksumh);
		// fprintf(fp,"h checksum %E\n", checksumh);

		free(testOutputi);
		free(testOutputc);
		free(testOutputh);
	}

	if (true)
	{
		float* testOutputdi;
		float* testOutputdh;
		float* testOutputdc;

		int biDirScale = (m_bidirectional ? 2 : 1);

		testOutputdi = (float*)malloc(m_inputSize * m_seqLength * m_miniBatch * sizeof(float));
		testOutputdh = (float*)malloc(m_hiddenSize * m_miniBatch * m_numLayers * biDirScale * sizeof(float));
		testOutputdc = (float*)malloc(m_hiddenSize * m_miniBatch * m_numLayers * biDirScale * sizeof(float));
		cudaErrCheck(cudaMemcpy(testOutputdi, m_dx, m_seqLength * m_miniBatch * m_inputSize * sizeof(float), cudaMemcpyDeviceToHost));
		if (m_dhx != NULL)
			cudaErrCheck(cudaMemcpy(testOutputdh, m_dhx, m_numLayers * m_hiddenSize * m_miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));
		if (m_dcx != NULL)
			if (m_RNNMode == CUDNN_LSTM)
				cudaErrCheck(cudaMemcpy(testOutputdc, m_dcx, m_numLayers * m_hiddenSize * m_miniBatch * biDirScale * sizeof(float), cudaMemcpyDeviceToHost));

		float checksumdi = 0.f;
		float checksumdh = 0.f;
		float checksumdc = 0.f;

		for (int m = 0; m < m_miniBatch; m++)
		{
			double localSumdi = 0;
			double localSumdh = 0;
			double localSumdc = 0;

			for (int j = 0; j < m_seqLength; j++)
			{
				for (int i = 0; i < m_inputSize; i++)
				{
					localSumdi += testOutputdi[j * m_miniBatch * m_inputSize + m * m_inputSize + i];
				}
			}

			for (int j = 0; j < m_numLayers * biDirScale; j++)
			{
				for (int i = 0; i < m_hiddenSize; i++)
				{
					localSumdh += testOutputdh[j * m_hiddenSize * m_miniBatch + m * m_hiddenSize + i];
					if (m_RNNMode == CUDNN_LSTM)
						localSumdc += testOutputdc[j * m_hiddenSize * m_miniBatch + m * m_hiddenSize + i];
				}
			}

			checksumdi += localSumdi;
			checksumdh += localSumdh;
			checksumdc += localSumdc;
		}

		printf("di checksum %E    ", checksumdi);
		// fprintf(fp,"di checksum %E    ", checksumdi);
		if (m_RNNMode == CUDNN_LSTM)
		{
			printf("dc checksum %E    ", checksumdc);
			// fprintf(fp,"dc checksum %E    ", checksumdc);
		}
		printf("dh checksum %E\n", checksumdh);
		// fprintf(fp,"dh checksum %E\n", checksumdh);

		free(testOutputdi);
		free(testOutputdh);
		free(testOutputdc);
	}

	if (true)
	{
		float* testOutputdw;
		testOutputdw = (float*)malloc(m_weightsSize);

		cudaErrCheck(cudaMemcpy(testOutputdw, m_dw, m_weightsSize, cudaMemcpyDeviceToHost));

		double checksumdw = 0.;

		for (int i = 0; i < m_weightsSize / sizeof(float); i++)
		{
			checksumdw += testOutputdw[i];
		}

		printf("dw checksum %E\n", checksumdw);
		// fprintf(fp,"dw checksum %E\n", checksumdw);

		free(testOutputdw);
	}
}

void RNN::calculate_FLOPS()
{
	if (m_RNNMode == CUDNN_RNN_RELU || m_RNNMode == CUDNN_RNN_TANH)
	{
		m_numMats = 2;
	}
	else if (m_RNNMode == CUDNN_LSTM)
	{
		m_numMats = 8;
	}
	else if (m_RNNMode == CUDNN_GRU)
	{
		m_numMats = 6;
	}

	// Calculate FLOPS
	printf("Forward: %3.0f GFLOPS\n", m_numMats * 2ull * (m_bidirectional ? 2 : 1) * m_hiddenSize * m_hiddenSize * m_seqLength * m_miniBatch * m_numLayers / (1e6 * m_timeForward));
	printf("Backward: %3.0f GFLOPS, ", m_numMats * 4ull * (m_bidirectional ? 2 : 1) * m_hiddenSize * m_hiddenSize * m_seqLength * m_miniBatch * m_numLayers / (1e6 * (m_timeBackward1 + m_timeBackward2)));
	printf("(%3.0f GFLOPS), ", m_numMats * 2ull * (m_bidirectional ? 2 : 1) * m_hiddenSize * m_hiddenSize * m_seqLength * m_miniBatch * m_numLayers / (1e6 * m_timeBackward1));
	printf("(%3.0f GFLOPS)\n", m_numMats * 2ull * (m_bidirectional ? 2 : 1) * m_hiddenSize * m_hiddenSize * m_seqLength * m_miniBatch * m_numLayers / (1e6 * m_timeBackward2));
}

void RNN::free_gpu_mem()
{
	// Make double-sure everything is finished before we copy for result checking.
	cudaDeviceSynchronize();

	if (m_algo == CUDNN_RNN_ALGO_PERSIST_DYNAMIC)
	{
		cudnnDestroyPersistentRNNPlan(m_rnnPlan);
	}

	cudaFree(m_x);
	cudaFree(m_hx);
	cudaFree(m_cx);
	cudaFree(m_y);
	cudaFree(m_hy);
	cudaFree(m_cy);
	cudaFree(m_dx);
	cudaFree(m_dhx);
	cudaFree(m_dcx);
	cudaFree(m_dy);
	cudaFree(m_dhy);
	cudaFree(m_dcy);
	cudaFree(m_workspace);
	cudaFree(m_reserveSpace);
	cudaFree(m_w);
	cudaFree(m_dw);
}

bool RNN::read_input_file()
{
	spdlog::info("start csv reading");
	io::CSVReader<18> input_stream(m_input_file);
	input_stream.read_header(io::ignore_missing_column | io::ignore_extra_column, "engine_id", "setting1", "setting2", "temp_lpc_outlet", "temp_hpc_outlet", "temp_lpt_outlet", "pressure_hpc_outlet", "physical_fan_speed", "physical_core_speed", "static_pressure_hpc_outlet", "fuel_flow_ration_ps30", "corrected_fan_speed", "corrected_core_speed", "bypass_ratio", "bleed_enthalpy", "hpt_collant_bleed", "lpt_coolant_bleed", "ttf");
    int count = 0;

	try
	{
        int engine_id;
		double setting1, setting2, temp_lpc_outlet, temp_hpc_outlet, temp_lpt_outlet, pressure_hpc_outlet, physical_fan_speed, physical_core_speed, static_pressure_hpc_outlet, fuel_flow_ration_ps30, corrected_fan_speed, corrected_core_speed, bypass_ratio, bleed_enthalpy, hpt_collant_bleed, lpt_coolant_bleed, ttf;
	    m_measurements_sums = std::make_unique<measurement>();
        measurement measurement_backup;

		while (input_stream.read_row(engine_id, setting1, setting2, temp_lpc_outlet, temp_hpc_outlet, temp_lpt_outlet, pressure_hpc_outlet, physical_fan_speed, physical_core_speed, static_pressure_hpc_outlet, fuel_flow_ration_ps30, corrected_fan_speed, corrected_core_speed, bypass_ratio, bleed_enthalpy, hpt_collant_bleed, lpt_coolant_bleed, ttf))
		{
            count += 1;
            
			std::unique_ptr<measurement> new_measurement = std::make_unique<measurement>(engine_id, setting1, setting2, temp_lpc_outlet, temp_hpc_outlet, temp_lpt_outlet, pressure_hpc_outlet, physical_fan_speed, physical_core_speed, static_pressure_hpc_outlet, fuel_flow_ration_ps30, corrected_fan_speed, corrected_core_speed, bypass_ratio, bleed_enthalpy, hpt_collant_bleed, lpt_coolant_bleed);

            // to enable sum at end of loop, not possible using the unique ptr from line above
            RNN::clone_measurement(&measurement_backup, new_measurement.get());
            
			if (engine_id <= 75)
			{
                std::unique_ptr<int> new_target = std::make_unique<int>(ttf);
                m_measurements_X_train.push_back(std::move(new_measurement));
                m_targets_Y_train.push_back(std::move(new_target));

			}
			else
			{
			    std::unique_ptr<int> new_target = std::make_unique<int>(ttf);
                m_measurements_X_test.push_back(std::move(new_measurement));
                m_targets_Y_test.push_back(std::move(new_target));
			}

            m_measurements_All.push_back(std::move(new_measurement));

            m_measurements_sums->sum(&measurement_backup);
		}
        m_measurements_means = m_measurements_sums->divide((double)count);
		
    	spdlog::info("std dev calc");
		m_measurements_std_devs = m_measurements_std_devs->std_dev(m_measurements_All, &m_measurements_means, count); // HIER IS DE ERROR
		spdlog::info("std dev calc done");
	}
	catch (const std::exception& e)
	{
		spdlog::error("Failure while reading csv:\n{}", e.what());

		return false;
	}
	

    spdlog::info("Loaded {}  measurement from {} \n  train: X {}, Y {} \n test: X {}, Y {}", count,  file_name(), m_measurements_X_train.size(), m_targets_Y_train.size(), m_measurements_X_test.size(), m_targets_Y_test.size());


    return true;
}

void RNN::standard_scale(std::vector<std::unique_ptr<measurement>>* input_measurements){
    float std_dev;
    float mean;

    measurement means = measurement(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);

    measurement curr_measurement_backup, previous_measurement;

    for (size_t i = 0; i < input_measurements->size(); i++)
        {
            std::unique_ptr<measurement>& curr_measurement = input_measurements->at(i);
            RNN::clone_measurement(&curr_measurement_backup, curr_measurement.get());

            if (i)
                // do stuff

            RNN::clone_measurement(&previous_measurement, &curr_measurement_backup);
        }

}

inline void RNN::clone_measurement(measurement* dst, measurement* src)
{
    memcpy(dst, src, sizeof(measurement));
}


const char* RNN::file_name()
{
	return m_input_file.filename().c_str();
}

RNN::~RNN()
{
	cudnnDestroy(m_Handle);
	spdlog::info("Destroyed cuDNN handle.");
}
