import json
import transformers
from pathlib import Path
from typing import Optional, List
from datasets import Split, NamedSplit
from dataclasses import dataclass, field, asdict


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    emulate_long_context: bool = False
    base_model: str = "llama-3.1"
    attn_func: str = "flash_attention_2"


CHUNKING_MODES = ["none", "overlap", "even"]


@dataclass
class DataArguments:
    train_file: Optional[str] = None
    val_file: Optional[str] = None
    test_file: Optional[str] = None
    task_names: List[str] = field(
        default_factory=list
    )  # used for establishing task instances
    accepted_tasks: List[str] = field(
        default_factory=list
    )  # used for filtering aceepted task instances
    removed_tasks: List[str] = field(
        default_factory=list
    )  # used for filtering removed task instances
    enable_cache: bool = True
    adaptive_masking: bool = False
    flexible_wording: str = ""
    replicate_num: int = 1
    balancing_replicates: bool = (
        False  # Cluster based on the queries and avoid the actual number of replicates exceeding the set replicate_num
    )
    use_ftfy: bool = False
    pretraining_chunking_mode: str = "overlap"
    role_play_in_pretraining: bool = False
    max_chunk_size_in_pretraining: int = 512  # The chunk size for pretraining
    min_overlap_size_in_pretraining: int = (
        64  # The minimum overlap size for pretraining
    )
    cache_path: str = ""  # Cache path for the training dataset
    remove_short_responses_le: int = -1  # If -1, no removal
    remove_long_queries_ge: int = -1  # If -1, no removal
    last_leap: bool = (
        False  # The final epoch should deprecate `flexible_wording` option
    )
    personalized: bool = False
    pre_clip: bool = False
    runtime_clip: bool = False
    runtime_ppl_filter: bool = False
    two_stage_combined: bool = False
    data_aug: bool = False
    perturb_degree: float = 0
    general_mixup: bool = False
    remove_sys: bool = False
    training_sample_max_length: int = 12000
    random_introduction_formatting: bool = False
    train_with_cpt: bool = False

    @property
    def data_format(self) -> str:
        return Path(self.train_file).suffix

    @property
    def data_files(self) -> dict[NamedSplit, str]:
        return {
            split: data_file
            for split, data_file in zip(
                [Split.TRAIN, Split.VALIDATION, Split.TEST],
                [self.train_file, self.val_file, self.test_file],
            )
            if data_file is not None
        }


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    curriculum_learning: bool = field(default=False)
    learn_step_by_step: List[str] = field(default_factory=lambda: ["others"])
    remove_unused_columns: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=16, metadata={"help": "How many bits to use."})
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    enable_prefix_tuning: bool = False
    num_virtual_tokens: int = 20
    is_pretraining: bool = False
    resume_from: str = ""
    use_lora: str = ""
    use_liger: bool = False

    def __post_init__(self):
        super().__post_init__()

        self.eval_on_start = True  # [https://discuss.huggingface.co/t/how-to-evaluate-before-first-training-step/18838/11]
        # self.load_best_model_at_end = True
        self.metric_for_best_model = "eval_loss"
        # self.save_steps = self.eval_steps


@dataclass
class ProcessArguments:
    manual_checking: bool = False
    llm_checking: bool = False
    min_query_token_num: int = 100
    retain_invalid: bool = False
    save_to: Optional[str] = None


def inherit_args(target_args, source_args, key):
    setattr(target_args, key, getattr(source_args, key))


def save_args_to_json(model_args, data_args, training_args, target_dir: str):
    """
    Save the provided arguments to a JSON file in the specified target directory.

    Parameters:
    model_args (ModelArguments): Model-related arguments.
    data_args (DataArguments): Data-related arguments.
    training_args (TrainingArguments): Training-related arguments.
    process_args (ProcessArguments): Process-related arguments.
    target_dir (str): Directory where the JSON file will be saved.
    """

    # Convert the dataclass instances to dictionaries
    model_args_dict = asdict(model_args)
    data_args_dict = asdict(data_args)
    training_args_dict = asdict(training_args)

    # Combine all dictionaries into one
    all_args = {
        "ModelArguments": model_args_dict,
        "DataArguments": data_args_dict,
        "TrainingArguments": training_args_dict,
    }

    # Create the target directory if it does not exist
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)

    # Define the JSON file path
    json_file_path = target_path / "args_config.json"

    # Save the combined dictionary to the JSON file
    with open(json_file_path, "w") as f:
        json.dump(all_args, f, indent=4)

    print(f"Arguments successfully saved to {json_file_path}")
