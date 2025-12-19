#!/bin/bash

# Model and API key
MODEL_NAME=
API_KEY=
TENSOR_PARALLEL_SIZE=

export CUDA_VISIBLE_DEVICES='0,1,2,3'  # Specify CUDA devices to use
export VLLM_RPC_TIMEOUT=20000
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Function to start the server
start_server() {
    python -m vllm.entrypoints.openai.api_server \
        --model $MODEL_NAME \
        --dtype auto \
        --api-key $API_KEY \
        --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
        --disable-log-requests \
        --enable-prefix-caching \
        --enforce-eager \
        --port 8000 \
        --max-num-batched-tokens 4096 \
        --guided-decoding-backend "lm-format-enforcer" \
        --served-model-name "Meta-Llama-3.1-70B-Instruct" \
        --gpu-memory-utilization 0.95
}

# Restart loop
while true; do
    echo "Starting server..."
    start_server
    EXIT_CODE=$?
    echo "Server exited with code $EXIT_CODE. Restarting in 5 seconds..."
    sleep 5
done
