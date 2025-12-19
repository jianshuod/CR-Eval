import os
from threading import Lock
from enum import Enum, auto
from src.utils.logging import logging
from src.configs import OAI_API_KEY, LOCAL_API_KEY, OAI_API_BASE

logger = logging.getLogger(__name__)


def get_cuda_visible_devices_count():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    if cuda_visible_devices:
        # Split the string by commas to get the list of device indices
        device_list = cuda_visible_devices.split(",")
        # Filter out empty strings if any
        device_list = [device.strip() for device in device_list if device.strip()]
        return len(device_list)
    else:
        # If CUDA_VISIBLE_DEVICES is not set, assume all devices are visible
        try:
            import torch

            return torch.cuda.device_count()
        except ImportError:
            # If torch is not available, return a default value
            return 0


MODEL_TPM_LIMITS = {"gpt-4o": 250_000, "gpt-4-turbo": 250_000, "gpt-3.5-turbo": 500_000}

MODEL_RPM_LIMITS = {"gpt-4o": 500, "gpt-4-turbo": 500, "gpt-3.5-turbo": 1_000}


MODEL_LIMITS = {
    "claude-instant-1": 100_000,
    "claude-2": 100_000,
    "claude-3-opus-20240229": 200_000,
    "claude-3-sonnet-20240229": 200_000,
    "claude-3-haiku-20240307": 200_000,
    "gpt-3.5-turbo": 16_385,
    "gpt-4o": 128_000,
    "gpt-4-turbo": 128_000,
    "gpt-4o-mini": 128_000,
    "llama-3-8b": 8_192,
    "llama-3-8b-instruct": 8_192,
    "llama-3-70b": 8_192,
    "llama-3-70b-instruct": 8_192,
    "glm-4-9b": 8_192,
    "glm-4-9b-chat": 131_072,
    "qwen2-7b": 131_072,
    "qwen2-7b-instruct": 32_768,
    "claude-3.5-sonnet": 200_000,
    "gemini-1.5-pro": 2_097_152,
    "gemini-1.5-pro-exp": 2_097_152,
    "gemma-2-27b": 4_096,
    "gemma-2-27b-it": 4_096,
    "gemma-2-9b": 4_096,
    "gemma-2-9b-it": 4_096,
    "yi-1.5-9b": 4_096,  # context size too small
    "yi-1.5-9b-chat": 4_096,
    "yi-1.5-34b": 4_096,
    "yi-1.5-34b-chat": 4_096,
    "cmd-r-plus": 4_096,
    "cmd-r-v01": 4_096,
    "qwen2-72b": 131_072,
    "qwen2-72b-instruct": 32_768,
    "mixtral-8x7b-instruct-v0.1": 32_768,
    "llama-3.1-8b": 131_072,
    "llama-3.1-8b-instruct": 131_072,
    "llama-3.1-70b": 131_072,
    "llama-3.1-70b-instruct": 131_072,
}


class ModelChoice(Enum):
    GPT_3_5_TURBO = auto()
    GPT_4_O = auto()
    GPT_4_TURBO = auto()
    GPT_4_O_MINI = auto()
    LLaMA_3_8B = auto()
    LLaMA_3_8B_INSTRUCT = auto()
    LLaMA_3_70B = auto()
    LLaMA_3_70B_INSTRUCT = auto()
    GLM_4_9B = auto()
    GLM_4_9B_CHAT = auto()
    QWEN2_7B = auto()
    QWEN2_7B_INSTRUCT = auto()
    QWEN2_72B = auto()
    QWEN2_72B_INSTRUCT = auto()
    YI_1_5_34B_CHAT = auto()
    YI_1_5_34B = auto()
    YI_1_5_9B = auto()
    YI_1_5_9B_CHAT = auto()
    CMD_R_PLUS = auto()
    CMD_R_V01 = auto()
    GEMMA_2_27B = auto()
    GEMMA_2_27B_IT = auto()
    GEMMA_2_9B = auto()
    GEMMA_2_9B_IT = auto()
    CLAUDE_3_5_SONNET = auto()
    GEMINI_1_5_PRO = auto()
    GEMINI_1_5_PRO_EXP = auto()
    Mixtral_8x7B_Instruct_v0_1 = auto()
    LLAMA_3_1_8B = auto()
    LLAMA_3_1_8B_INSTRUCT = auto()
    LLAMA_3_1_70B = auto()
    LLAMA_3_1_70B_INSTRUCT = auto()
    LOCAL_CHECKPOINT = auto()
    OAI_ENDPOINT = auto()

    server_num: int = 0
    local_checkpoint_path: str = ""

    def get_model_string(self):
        if self == ModelChoice.GPT_3_5_TURBO:
            return "gpt-3.5-turbo"
        elif self == ModelChoice.GPT_4_O:
            return "gpt-4o"
        elif self == ModelChoice.GPT_4_TURBO:
            return "gpt-4-turbo"
        elif self == ModelChoice.GPT_4_O_MINI:
            return "gpt-4o-mini"
        elif self == ModelChoice.LLaMA_3_8B:
            return "llama-3-8b"
        elif self == ModelChoice.LLaMA_3_8B_INSTRUCT:
            return "llama-3-8b-instruct"
        elif self == ModelChoice.LLaMA_3_70B:
            return "llama-3-70b"
        elif self == ModelChoice.LLaMA_3_70B_INSTRUCT:
            return "llama-3-8b-instruct"
        elif self == ModelChoice.CLAUDE_3_5_SONNET:
            return "claude-3-5-sonnet"
        elif self == ModelChoice.GEMINI_1_5_PRO:
            return "gemini-1.5-pro"
        elif self == ModelChoice.GEMINI_1_5_PRO_EXP:
            return "gemini-1.5-pro-exp"
        elif self == ModelChoice.GLM_4_9B:
            return "glm-4-9b"
        elif self == ModelChoice.GLM_4_9B_CHAT:
            return "glm-4-9b-chat"
        elif self == ModelChoice.QWEN2_7B:
            return "qwen2-7b"
        elif self == ModelChoice.QWEN2_7B_INSTRUCT:
            return "qwen2-7b-instruct"
        elif self == ModelChoice.QWEN2_72B:
            return "qwen2-72b"
        elif self == ModelChoice.QWEN2_72B_INSTRUCT:
            return "qwen2-72b-instruct"
        elif self == ModelChoice.YI_1_5_34B_CHAT:
            return "yi-1.5-34b-chat"
        elif self == ModelChoice.YI_1_5_34B:
            return "yi-1.5-34b"
        elif self == ModelChoice.YI_1_5_9B:
            return "yi-1.5-9b"
        elif self == ModelChoice.YI_1_5_9B_CHAT:
            return "yi-1.5-9b-chat"
        elif self == ModelChoice.CMD_R_PLUS:
            return "cmd-r-plus"
        elif self == ModelChoice.CMD_R_V01:
            return "cmd-r-v01"
        elif self == ModelChoice.GEMMA_2_27B:
            return "gemma-2-27b"
        elif self == ModelChoice.GEMMA_2_27B_IT:
            return "gemma-2-27b-it"
        elif self == ModelChoice.GEMMA_2_9B:
            return "gemma-2-9b"
        elif self == ModelChoice.GEMMA_2_9B_IT:
            return "gemma-2-9b-it"
        elif self == ModelChoice.Mixtral_8x7B_Instruct_v0_1:
            return "mixtral-8x7b-instruct-v0.1"
        elif self == ModelChoice.LLAMA_3_1_8B:
            return "llama-3.1-8b"
        elif self == ModelChoice.LLAMA_3_1_8B_INSTRUCT:
            return "llama-3.1-8b-instruct"
        elif self == ModelChoice.LLAMA_3_1_70B:
            return "llama-3.1-70b"
        elif self == ModelChoice.LLAMA_3_1_70B_INSTRUCT:
            return "llama-3.1-70b-instruct"
        elif self == ModelChoice.LOCAL_CHECKPOINT:
            return self.local_checkpoint_path
        elif self == ModelChoice.OAI_ENDPOINT:
            return "oai-endpoint"
        else:
            raise ValueError(f"Unknown model choice")

    def is_chat(self):
        if self == ModelChoice.GPT_3_5_TURBO:
            return True
        elif self == ModelChoice.GPT_4_O:
            return True
        elif self == ModelChoice.GPT_4_TURBO:
            return True
        elif self == ModelChoice.GPT_4_O_MINI:
            return True
        elif self == ModelChoice.GEMINI_1_5_PRO:
            return True
        elif self == ModelChoice.GEMINI_1_5_PRO_EXP:
            return True
        elif self == ModelChoice.LLaMA_3_8B:
            return False
        elif self == ModelChoice.LLaMA_3_8B_INSTRUCT:
            return True
        elif self == ModelChoice.LLaMA_3_70B:
            return False
        elif self == ModelChoice.LLaMA_3_70B_INSTRUCT:
            return True
        elif self == ModelChoice.CLAUDE_3_5_SONNET:
            return True
        elif self == ModelChoice.GLM_4_9B:
            return False
        elif self == ModelChoice.GLM_4_9B_CHAT:
            return True
        elif self == ModelChoice.QWEN2_7B:
            return False
        elif self == ModelChoice.QWEN2_7B_INSTRUCT:
            return True
        elif self == ModelChoice.QWEN2_72B:
            return False
        elif self == ModelChoice.QWEN2_72B_INSTRUCT:
            return True
        elif self == ModelChoice.YI_1_5_34B_CHAT:
            return True
        elif self == ModelChoice.YI_1_5_34B:
            return False
        elif self == ModelChoice.YI_1_5_9B:
            return False
        elif self == ModelChoice.YI_1_5_9B_CHAT:
            return True
        elif self == ModelChoice.CMD_R_PLUS:
            return True
        elif self == ModelChoice.CMD_R_V01:
            return True
        elif self == ModelChoice.GEMMA_2_27B:
            return False
        elif self == ModelChoice.GEMMA_2_27B_IT:
            return True
        elif self == ModelChoice.GEMMA_2_9B:
            return False
        elif self == ModelChoice.GEMMA_2_9B_IT:
            return True
        elif self == ModelChoice.Mixtral_8x7B_Instruct_v0_1:
            return True
        elif self == ModelChoice.LLAMA_3_1_8B:
            return False
        elif self == ModelChoice.LLAMA_3_1_8B_INSTRUCT:
            return True
        elif self == ModelChoice.LLAMA_3_1_70B:
            return False
        elif self == ModelChoice.LLAMA_3_1_70B_INSTRUCT:
            return True
        elif self == ModelChoice.LOCAL_CHECKPOINT:
            return False
        elif self == ModelChoice.OAI_ENDPOINT:
            return True
        else:
            raise ValueError(f"Unknown model choice")

    def get_vllm_configs(self):
        if self in (
            ModelChoice.GPT_3_5_TURBO,
            ModelChoice.GPT_4_O,
            ModelChoice.GPT_4_TURBO,
            ModelChoice.CLAUDE_3_5_SONNET,
            ModelChoice.GPT_4_O_MINI,
            ModelChoice.GEMINI_1_5_PRO,
            ModelChoice.GEMINI_1_5_PRO_EXP,
            ModelChoice.OAI_ENDPOINT,
        ):
            return None
        elif self in (
            ModelChoice.LLaMA_3_8B,
            ModelChoice.LLaMA_3_8B_INSTRUCT,
            ModelChoice.GLM_4_9B,
            ModelChoice.GLM_4_9B_CHAT,
            ModelChoice.QWEN2_7B,
            ModelChoice.QWEN2_7B_INSTRUCT,
            ModelChoice.YI_1_5_9B,
            ModelChoice.YI_1_5_9B_CHAT,
            ModelChoice.GEMMA_2_9B,
            ModelChoice.GEMMA_2_9B_IT,
            ModelChoice.LLAMA_3_1_8B,
            ModelChoice.LLAMA_3_1_8B_INSTRUCT,
        ):
            return {
                "tensor_parallel_size": 1,
            }
        elif self in (
            ModelChoice.GEMMA_2_27B,
            ModelChoice.GEMMA_2_27B_IT,
            ModelChoice.YI_1_5_34B,
            ModelChoice.YI_1_5_34B_CHAT,
            ModelChoice.CMD_R_V01,
            ModelChoice.Mixtral_8x7B_Instruct_v0_1,
        ):
            return {
                "tensor_parallel_size": 2,
            }
        elif self in (
            ModelChoice.LLaMA_3_70B,
            ModelChoice.LLaMA_3_70B_INSTRUCT,
            ModelChoice.QWEN2_72B,
            ModelChoice.QWEN2_72B_INSTRUCT,
            ModelChoice.CMD_R_PLUS,
            ModelChoice.LLAMA_3_1_70B,
            ModelChoice.LLAMA_3_1_70B_INSTRUCT,
        ):
            return {
                "tensor_parallel_size": 4,
            }
        elif self in (ModelChoice.LOCAL_CHECKPOINT,):
            return {
                "tensor_parallel_size": self.server_num,
            }
        else:
            raise ValueError(f"Unknown model choice")

    def get_base_url(self):
        if self in (
            ModelChoice.GPT_3_5_TURBO,
            ModelChoice.GPT_4_O,
            ModelChoice.GPT_4_TURBO,
            ModelChoice.CLAUDE_3_5_SONNET,
            ModelChoice.GPT_4_O_MINI,
            ModelChoice.GEMINI_1_5_PRO,
            ModelChoice.GEMINI_1_5_PRO_EXP,
            ModelChoice.OAI_ENDPOINT,
        ):
            return f"{OAI_API_BASE}/v1"
        elif self in (
            ModelChoice.LLaMA_3_8B,
            ModelChoice.LLaMA_3_8B_INSTRUCT,
            ModelChoice.GLM_4_9B,
            ModelChoice.GLM_4_9B_CHAT,
            ModelChoice.QWEN2_7B,
            ModelChoice.QWEN2_7B_INSTRUCT,
            ModelChoice.YI_1_5_9B,
            ModelChoice.YI_1_5_9B_CHAT,
            ModelChoice.GEMMA_2_9B,
            ModelChoice.GEMMA_2_9B_IT,
            ModelChoice.GEMMA_2_27B,
            ModelChoice.GEMMA_2_27B_IT,
            ModelChoice.YI_1_5_34B,
            ModelChoice.YI_1_5_34B_CHAT,
            ModelChoice.CMD_R_V01,
            ModelChoice.Mixtral_8x7B_Instruct_v0_1,
            ModelChoice.LLaMA_3_70B,
            ModelChoice.LLaMA_3_70B_INSTRUCT,
            ModelChoice.QWEN2_72B,
            ModelChoice.QWEN2_72B_INSTRUCT,
            ModelChoice.CMD_R_PLUS,
            ModelChoice.LLAMA_3_1_8B,
            ModelChoice.LLAMA_3_1_8B_INSTRUCT,
            ModelChoice.LLAMA_3_1_70B,
            ModelChoice.LLAMA_3_1_70B_INSTRUCT,
            ModelChoice.LOCAL_CHECKPOINT,
        ):
            candidate_url_list = [
                f"http://localhost:{8000 + i}/v1" for i in range(self.server_num)
            ]
            with llama_index_lock:
                url = candidate_url_list[llama_index]
                update_llama_index(self.server_num)
            return url
        else:
            raise ValueError(f"Unknown model choice")

    def get_max_workers(self, server_num):

        self.server_num = server_num

        if self in (
            ModelChoice.GPT_3_5_TURBO,
            ModelChoice.GPT_4_O,
            ModelChoice.GPT_4_TURBO,
            ModelChoice.CLAUDE_3_5_SONNET,
            ModelChoice.GPT_4_O_MINI,
            ModelChoice.GEMINI_1_5_PRO,
            ModelChoice.GEMINI_1_5_PRO_EXP,
            ModelChoice.OAI_ENDPOINT,
        ):
            return 128
        elif self in (
            ModelChoice.LLaMA_3_8B,
            ModelChoice.LLaMA_3_8B_INSTRUCT,
            ModelChoice.GLM_4_9B,
            ModelChoice.GLM_4_9B_CHAT,
            ModelChoice.QWEN2_7B,
            ModelChoice.QWEN2_7B_INSTRUCT,
            ModelChoice.YI_1_5_9B,
            ModelChoice.YI_1_5_9B_CHAT,
            ModelChoice.GEMMA_2_9B,
            ModelChoice.GEMMA_2_9B_IT,
            ModelChoice.GEMMA_2_27B,
            ModelChoice.GEMMA_2_27B_IT,
            ModelChoice.YI_1_5_34B,
            ModelChoice.YI_1_5_34B_CHAT,
            ModelChoice.CMD_R_V01,
            ModelChoice.Mixtral_8x7B_Instruct_v0_1,
            ModelChoice.LLaMA_3_70B,
            ModelChoice.LLaMA_3_70B_INSTRUCT,
            ModelChoice.QWEN2_72B,
            ModelChoice.QWEN2_72B_INSTRUCT,
            ModelChoice.CMD_R_PLUS,
            ModelChoice.LLAMA_3_1_8B,
            ModelChoice.LLAMA_3_1_8B_INSTRUCT,
            ModelChoice.LLAMA_3_1_70B,
            ModelChoice.LLAMA_3_1_70B_INSTRUCT,
            ModelChoice.LOCAL_CHECKPOINT,
        ):
            return 32 * server_num
        else:
            raise ValueError(f"Unknown model choice")

    def get_api_key(self):
        if self in (
            ModelChoice.GPT_3_5_TURBO,
            ModelChoice.GPT_4_O,
            ModelChoice.GPT_4_TURBO,
            ModelChoice.CLAUDE_3_5_SONNET,
            ModelChoice.GPT_4_O_MINI,
            ModelChoice.GEMINI_1_5_PRO,
            ModelChoice.GEMINI_1_5_PRO_EXP,
            ModelChoice.OAI_ENDPOINT,
        ):
            return OAI_API_KEY
        elif self in (
            ModelChoice.LLaMA_3_8B,
            ModelChoice.LLaMA_3_8B_INSTRUCT,
            ModelChoice.GLM_4_9B,
            ModelChoice.GLM_4_9B_CHAT,
            ModelChoice.LLaMA_3_70B,
            ModelChoice.LLaMA_3_70B_INSTRUCT,
            ModelChoice.QWEN2_7B,
            ModelChoice.QWEN2_7B_INSTRUCT,
            ModelChoice.YI_1_5_9B,
            ModelChoice.YI_1_5_9B_CHAT,
            ModelChoice.GEMMA_2_9B,
            ModelChoice.GEMMA_2_9B_IT,
            ModelChoice.GEMMA_2_27B,
            ModelChoice.GEMMA_2_27B_IT,
            ModelChoice.YI_1_5_34B,
            ModelChoice.YI_1_5_34B_CHAT,
            ModelChoice.CMD_R_V01,
            ModelChoice.CMD_R_PLUS,
            ModelChoice.QWEN2_72B,
            ModelChoice.QWEN2_72B_INSTRUCT,
            ModelChoice.Mixtral_8x7B_Instruct_v0_1,
            ModelChoice.LLAMA_3_1_8B,
            ModelChoice.LLAMA_3_1_8B_INSTRUCT,
            ModelChoice.LLAMA_3_1_70B,
            ModelChoice.LLAMA_3_1_70B_INSTRUCT,
            ModelChoice.LOCAL_CHECKPOINT,
        ):
            return LOCAL_API_KEY
        else:
            raise ValueError(f"Unknown model choice")

    @classmethod
    def from_string(cls, model_name: str):
        # Create a dictionary mapping model strings to ModelChoice members
        model_map = {
            "gpt-3.5-turbo": cls.GPT_3_5_TURBO,
            "gpt-4o": cls.GPT_4_O,
            "gpt-4-turbo": cls.GPT_4_TURBO,
            "gpt-4o-mini": cls.GPT_4_O_MINI,
            "llama-3-8b": cls.LLaMA_3_8B,
            "llama-3-8b-instruct": cls.LLaMA_3_8B_INSTRUCT,
            "llama-3-70b": cls.LLaMA_3_70B,
            "llama-3-70b-instruct": cls.LLaMA_3_70B_INSTRUCT,
            "claude-3.5-sonnet": cls.CLAUDE_3_5_SONNET,
            "gemini-1.5-pro": cls.GEMINI_1_5_PRO,
            "gemini-1.5-pro-exp": cls.GEMINI_1_5_PRO_EXP,
            "glm-4-9b": cls.GLM_4_9B,
            "glm-4-9b-chat": cls.GLM_4_9B_CHAT,
            "qwen2-7b": cls.QWEN2_7B,
            "qwen2-7b-instruct": cls.QWEN2_7B_INSTRUCT,
            "qwen2-72b": cls.QWEN2_72B,
            "qwen2-72b-instruct": cls.QWEN2_72B_INSTRUCT,
            "yi-1.5-34b-chat": cls.YI_1_5_34B_CHAT,
            "yi-1.5-34b": cls.YI_1_5_34B,
            "yi-1.5-9b": cls.YI_1_5_9B,
            "yi-1.5-9b-chat": cls.YI_1_5_9B_CHAT,
            "cmd-r-plus": cls.CMD_R_PLUS,
            "cmd-r-v01": cls.CMD_R_V01,
            "gemma-2-27b": cls.GEMMA_2_27B,
            "gemma-2-27b-it": cls.GEMMA_2_27B_IT,
            "gemma-2-9b": cls.GEMMA_2_9B,
            "gemma-2-9b-it": cls.GEMMA_2_9B_IT,
            "mixtral-8x7b-instruct-v0.1": cls.Mixtral_8x7B_Instruct_v0_1,
            "llama-3.1-8b": cls.LLAMA_3_1_8B,
            "llama-3.1-8b-instruct": cls.LLAMA_3_1_8B_INSTRUCT,
            "llama-3.1-70b": cls.LLAMA_3_1_70B,
            "llama-3.1-70b-instruct": cls.LLAMA_3_1_70B_INSTRUCT,
        }
        # Return the corresponding ModelChoice member or raise an error if not found
        if model_name in model_map:
            return model_map[model_name]
        elif os.path.exists(model_name):
            cls.LOCAL_CHECKPOINT.local_checkpoint_path = model_name
            return cls.LOCAL_CHECKPOINT
        elif model_name.startswith("oai-endpoint"):
            return cls.OAI_ENDPOINT
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def get_tokenizer_id(self):
        if self == ModelChoice.LLaMA_3_8B:
            return "LLM-Research/Meta-Llama-3-8B"
        elif self == ModelChoice.LLaMA_3_8B_INSTRUCT:
            return "LLM-Research/Meta-Llama-3-8B-Instruct"
        elif self == ModelChoice.LLaMA_3_70B:
            return "LLM-Research/Meta-Llama-3-70B"
        elif self == ModelChoice.LLaMA_3_70B_INSTRUCT:
            return "LLM-Research/Meta-Llama-3-70B-Instruct"
        elif self == ModelChoice.LLAMA_3_1_8B:
            return "LLM-Research/Meta-Llama-3.1-8B"
        elif self == ModelChoice.LLAMA_3_1_8B_INSTRUCT:
            return "LLM-Research/Meta-Llama-3.1-8B-Instruct"
        elif self == ModelChoice.LLAMA_3_1_70B:
            return "LLM-Research/Meta-Llama-3.1-70B"
        elif self == ModelChoice.LLAMA_3_1_70B_INSTRUCT:
            return "LLM-Research/Meta-Llama-3.1-70B-Instruct"
        elif self == ModelChoice.GLM_4_9B:
            return "ZhipuAI/glm-4-9b"
        elif self == ModelChoice.GLM_4_9B_CHAT:
            return "ZhipuAI/glm-4-9b-chat"
        elif self == ModelChoice.QWEN2_7B:
            return "qwen/Qwen2-7B"
        elif self == ModelChoice.QWEN2_7B_INSTRUCT:
            return "qwen/Qwen2-7B-Instruct"
        elif self == ModelChoice.QWEN2_72B:
            return "qwen/Qwen2-72B"
        elif self == ModelChoice.QWEN2_72B_INSTRUCT:
            return "qwen/Qwen2-72B-Instruct"
        elif self == ModelChoice.YI_1_5_34B_CHAT:
            return "01ai/Yi-1.5-34B-Chat"
        elif self == ModelChoice.YI_1_5_34B:
            return "01ai/Yi-1.5-34B"
        elif self == ModelChoice.YI_1_5_9B:
            return "01ai/Yi-1.5-9B"
        elif self == ModelChoice.YI_1_5_9B_CHAT:
            return "01ai/Yi-1.5-9B-Chat"
        elif self == ModelChoice.CMD_R_PLUS:
            return "AI-ModelScope/c4ai-command-r-plus"
        elif self == ModelChoice.CMD_R_V01:
            return "AI-ModelScope/c4ai-command-r-v01"
        elif self == ModelChoice.GEMMA_2_27B:
            return "LLM-Research/gemma-2-27b"
        elif self == ModelChoice.GEMMA_2_27B_IT:
            return "LLM-Research/gemma-2-27b-it"
        elif self == ModelChoice.GEMMA_2_9B:
            return "LLM-Research/gemma-2-9b"
        elif self == ModelChoice.GEMMA_2_9B_IT:
            return "LLM-Research/gemma-2-9b-it"
        elif self == ModelChoice.Mixtral_8x7B_Instruct_v0_1:
            return "AI-ModelScope/Mixtral-8x7B-Instruct-v0.1"
        elif self == ModelChoice.LOCAL_CHECKPOINT:
            return self.local_checkpoint_path
        else:
            raise ValueError(f"Unknown model choice")


model_name_conversion = {
    "llama-3-8b": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-8b-instruct": "meta-llama/Meta-Llama-3-8B-Instruct",
    "llama-3-70b": "meta-llama/Meta-Llama-3-70B",
    "llama-3-70b-instruct": "meta-llama/Meta-Llama-3-70B-Instruct",
    "llama-3.1-8b": "meta-llama/Meta-Llama-3.1-8B",
    "llama-3.1-8b-instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "llama-3.1-70b": "meta-llama/Meta-Llama-3.1-70B",
    "llama-3.1-70b-instruct": "meta-llama/Meta-Llama-3.1-70B-Instruct",
    "glm-4-9b": "THUDM/glm-4-9b",
    "glm-4-9b-chat": "THUDM/glm-4-9b-chat",
    "gpt-4o": "gpt-4o",
    "claude-3.5-sonnet": "claude-3-5-sonnet",
    "gemini-1.5-pro": "gemini-1.5-pro",
    "qwen2-7b-instruct": "Qwen/Qwen2-7B-Instruct",
    "qwen2-72b-instruct": "Qwen/Qwen2-72B-Instruct"
}


# External variables and functions
llama_index = 0
llama_index_lock = Lock()
cuda_device_count = get_cuda_visible_devices_count()


def get_llama_index():
    global llama_index
    return llama_index


def update_llama_index(server_num):
    global llama_index
    llama_index = (llama_index + 1) % server_num


class TaskType(Enum):
    """Task type"""

    IMAGE = auto()
    TEXT = auto()