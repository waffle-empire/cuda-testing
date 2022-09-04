#include "rnn.hpp"
#include <cuda.h>
#include <stdio.h>
#include <spdlog/spdlog.h>
#include <filesystem>
#include <fstream>
#include <iostream>

int main(int argc, char** argv)
{
    int seqLength;
    int numLayers;
    int hiddenSize;
    int miniBatch;
    float dropout;
    bool bidirectional;
    int mode;
    int algo;

    if (argc == 9) {
        seqLength = atoi(argv[1]);
        numLayers = atoi(argv[2]);
        hiddenSize = atoi(argv[3]);
        miniBatch = atoi(argv[4]);
        dropout = atoi(argv[5]);
        bidirectional = atoi(argv[6]);
        mode = atoi(argv[7]);
        algo = atoi(argv[8]);
    }
    else {
        spdlog::info("Usage:");
        spdlog::info("./RNN <seqLength> <numLayers> <hiddenSize> <miniBatch> <dropout> <bidirectional> <mode> <algo>");
        spdlog::info("Modes: 0 = RNN_RELU, 1 = RNN_TANH, 2 = LSTM, 3 = GRU");
        return 1;
    }

    std::filesystem::path input_file = "../test3/failures.csv";
    RNN* rnn{new RNN(input_file, seqLength, numLayers, hiddenSize, miniBatch, dropout, bidirectional, mode, algo)};

    rnn->read_input_file();

    rnn->setup_input_output();
    rnn->setup_tensor_descriptors();
    rnn->setup_dropout_descriptor();
    rnn->setup_rnn_descriptor();
    rnn->setup_parameters();
    rnn->setup_workspace_memory();
    rnn->setup_weights_and_inputs();
    // *********************************************************************************************************
    // At this point all of the setup is done. We now need to pass through the RNN.
    // *********************************************************************************************************
    rnn->train();

    rnn->print_checksums();
    rnn->calculate_FLOPS();
    rnn->free_gpu_mem();

    delete rnn;
    rnn = nullptr;

    return 0;
}