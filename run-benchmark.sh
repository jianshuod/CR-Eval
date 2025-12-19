#!/bin/bash

output_dir="" # Directory to save the results
input_jsonl="" # Path to the input jsonl file of the subset of CR-eval

############# FILL CR ##################

test_model="Qwen/Qwen3-32B"
python run_live.py --num_worker 16 --model_name "$test_model" --task_name "fill-cr" --max_model_length 20000 --from_jsonl "$input_jsonl" --output_dir "$output_dir" --eval_funcs_str "gpt-score-fill-cr-v5" --eval_configs_for_gpt_scorer "only-scoring" --eval_repetition 1 --version "llama3" --k_trials 10 --gen_conf '{"max_response_token_num":800, "temperature":0.8, "top_p":0.95}'  --is_third_party_chat True --vllm_config '{"tensor_parallel_size":2}'