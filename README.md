# CRiticLM

This repository contains the official implementation of ***CRiticLM***: Fine-Tuned Large Language Models Are Weakness Analysts of Cellular Network Specifications

## System Requirements

This project successfully runs on:

- OS: `Ubuntu 20.04.5 LTS`
- GPU: `NVIDIA H800 80GB`
- CUDA Version: `11.8`
- Python Version: `3.12`

## Setup

### Install Requirements

```bash
# For conda users
conda env create -f environment.yml

# For pip users
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Download Datasets
Datasets are provided in the directory `assets/benchmarks`


### Set Environment Variables

Rename `.env.template` to `.env` and fill in the required environment variables.


## Codebase Components


### Data collection and processing

`src/collect`, `src/make_data` and `src/process` handle the collection and preprocessing of data from various sources. This component facilitates the automated downloading of Change Requests (CRs) and other relevant documents from 3GPP FTP servers. The collected data undergoes preprocessing steps to ensure it is in a suitable format for further analysis, which includes filtering, validation, and structuring.

### CR-Eval

We provide a easy-to-use bash script `run-benchmark.sh`, including a typical template for evaluating models in CR-Eval.

`src/eval`, `src/analysis` and `run_live.py` conduct live inference sessions on 3GPP Change Request, which automates the process of fetching target CRs, building up retrieval index, generating prompt and judging output. `run_live.py` is the main entry point for this component.

```bash
python run_live.py \
    --model_name MODEL_NAME \
    --task_name TASK_NAME \
    --cr_indices CR_PATHS \
    --output_dir OUTPUT_DIR \
    [optional arguments]
```

#### Arguments

Required Arguments
- `--model_name`: Model name or local path to use for inference
- `--task_name`: Task name to execute (choices: available tasks in the system)
- `--cr_indices`: One or more CR paths, each starting with either "local:" or "ftp:"
- `--output_dir`: Directory to save the output JSONL file (default: "./live_outputs")

Model Configuration
- `--using_lora`: Path to LoRA adapter (optional)
- `--version`: Prompt version for local checkpoints (default: "llama3")
- `--max_context_length`: Maximum context length for input text (default: 16000)
- `--max_model_length`: Maximum model length for input text
- `--vllm_config`: Dictionary of VLLM configuration parameters
- `--gen_conf`: Dictionary of generation parameters

Input Processing
- `--chunking_mode`: Chunking mode for input text (choices: available in CHUNKING_FUNCTIONS, default: "section")
- `--dataset_dir`: Path to dataset directory containing 3GPP CR files
- `--from_jsonl`: Path to JSONL file containing conversations
- `--retrieval_mode`: Mode for fetching spec statements (choices: "oracle", "bm25", default: "oracle")
- `--document_encoding_func_str`: Function for encoding the spec (default: "file_name_and_contents")

Prompt and System Configuration
- `--system_suffix`: Additional system instruction for LLM workers
- `--assistant_prefix`: Additional assistant prefix for completion models
- `--use_my_prompt`: Custom prompt for the model
- `--personalized`: Whether to use personalization (default: False)
- `--is_third_party_chat`: Use third-party chat model templates (default: False)
- `--add_shots_from_jsonl`: Load example shots from a JSONL file

Evaluation Configuration
- `--eval_funcs_str`: Evaluation functions for scoring model output
- `--eval_configs_for_gpt_scorer`: Configuration choices for GPT scorer (default: ["only-scoring"])
- `--eval_repetition`: Number of evaluation repetitions (default: 1)
- `--k_trials`: Number of trials for pass@k metric (default: 1)

Runtime Configuration
- `--num_worker`: Maximum workers for querying LLM endpoint (default: -1)
- `--run_name`: Custom name for the run

#### Example Usage

An example usage of `run_live.py` is provided in `run-benchmark.sh`.

### Notes
- Paths starting with "local:" refer to local files
- Paths starting with "ftp:" will download from the 3GPP server
- Multiple CR indices can be provided for batch processing
- The output will be saved in JSONL format with evaluation metrics


### Training

`src/train` contains the core training scripts and utilities for training and fine-tuning models in CRitic.



<!-- 
## Citation
If you find this work useful in your research, please consider citing the following paper:
```
``` -->
