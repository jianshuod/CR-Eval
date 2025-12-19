#!/bin/bash

# Define model and API key
MODEL_NAME=
API_KEY=
TENSOR_PARALLEL_SIZE=
LORA_MODULES=

# Set environment variables
export CUDA_VISIBLE_DEVICES='0,1,2,3'  # Specify CUDA devices to use

# Start the first server with the first CUDA device
CUDA_VISIBLE_DEVICES='0' python -m vllm.entrypoints.openai.api_server \
    --model $MODEL_NAME \
    --enable_lora \
    --lora-modules lora=$LORA_MODULES \
    --dtype auto \
    --api-key $API_KEY \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --disable-log-requests \
    --port 8000
