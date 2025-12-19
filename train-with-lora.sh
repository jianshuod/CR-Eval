#!/bin/bash

PROMPT_VERSION="llama3"
LOG_DIR=""
TRAIN_FILE=""
VAL_FILE=""

export PYTHONPATH=$(pwd):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_SOCKET_IFNAME=eth0
export TORCH_NCCL_ENABLE_MONITORING=0 # To avoid killing the process when runtime processing the data
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

export CUDA_LAUNCH_BLOCKING=1

# Function to check GPU utilization
check_gpu_free() {
    echo "Checking GPU memory utilization..."
    
    # Get the memory utilization percentage for all GPUs
    GPU_MEMORY_UTILS=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
    echo "GPU memory utilization (used/total):"
    echo "$GPU_MEMORY_UTILS"
    
    # Iterate through each GPU memory utilization
    while IFS=',' read -r used_mem total_mem; do
        mem_util=$((100 * used_mem / total_mem))
        echo "GPU memory utilization: $mem_util%"
        
        # Check if any GPU memory utilization is 10% or higher
        if [[ $mem_util -ge 10 ]]; then
            echo "GPU memory is busy (utilization >= 10%)"
            # If any GPU memory is busy (utilization >= 10%), return 1 (indicating busy)
            return 1
        fi
    done <<< "$GPU_MEMORY_UTILS"
    
    echo "All GPUs are free (memory utilization < 10%)"
    # If all GPU memory is free (utilization < 10%), return 0 (indicating free)
    return 0
}


# Loop to check GPU status every few minutes (e.g., every 5 minutes)
while true; do
    # Call the function to check if GPUs are free
    if check_gpu_free; then
        echo "GPUs are free, starting the training script..."


        export WANDB_PROJECT="CRitic" 

        MODEL_VERSION="Meta-Llama-3___1-8B-Instruct"
        RUN_NAME="fill-cr-stage3"

        echo $RUN_NAME

        deepspeed src/train/train_mem_nightly.py \
            --deepspeed ./configs/custom_zero2_for_stage3.json \
            --lora_enable True --lora_r 256 --lora_alpha 512 \
            --model_name_or_path ../assets/checkpoints/$MODEL_VERSION \
            --version $PROMPT_VERSION \
            --bf16 True \
            --train_file $TRAIN_FILE \
            --val_file $VAL_FILE \
            --output_dir ../assets/checkpoints/$RUN_NAME \
            --num_train_epochs 1 \
            --per_device_train_batch_size 2 \
            --per_device_eval_batch_size 2 \
            --gradient_accumulation_steps 16 \
            --eval_strategy "steps" \
            --eval_steps 20 \
            --save_strategy "no" \
            --save_total_limit 1 \
            --save_steps 50 \
            --learning_rate 5e-5 \
            --weight_decay 0.0 \
            --warmup_ratio 0.03 \
            --lr_scheduler_type "cosine_with_min_lr" \
            --lr_scheduler_kwargs '{"min_lr_rate":0.05}' \
            --logging_steps 1 \
            --tf32 True \
            --model_max_length 12000 \
            --gradient_checkpointing True \
            --gradient_checkpointing_kwargs '{"use_reentrant": true}' \
            --report_to wandb \
            --run_name $RUN_NAME \
            --replicate_num 5 \
            --balancing_replicates True \
            --dataloader_num_workers 16 \
            --base_model "llama-3.1" \
            --enable_cache True \
            --seed 1014 >$LOG_DIR/$RUN_NAME.txt 2>&1

        # Exit the loop after starting the training script
        break
    else
        echo "GPUs are busy, checking again in 5 minutes..."
        # Wait for 5 minutes before checking again
        sleep $((5 * 60))
    fi
done
