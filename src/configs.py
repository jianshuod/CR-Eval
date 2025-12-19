import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MAGIC_NUM = "0x1234ABCD"

HUGGINGFACE_HUB_CACHE = os.getenv("HUGGINGFACE_HUB_CACHE")

# Retrieve environment variables
PROJECT_DIR = Path(__file__).parent.parent.as_posix()
ASSET_DIR = f"{PROJECT_DIR}/assets"
CACHE_DIR = f"{PROJECT_DIR}/caches"
LOG_DIR = f"{PROJECT_DIR}/logs"

# Hyper-parameters
RESPONSE_SPACE = int(os.getenv("RESPONSE_SPACE"))
MAX_CONTEXT_RATE = float(os.getenv("MAX_CONTEXT_RATE"))


# OpenAI API
OAI_API_KEY = os.getenv("OAI_API_KEY")
OAI_API_BASE = os.getenv("OAI_API_BASE")
LOCAL_API_KEY = os.getenv("LOCAL_API_KEY")


# Cache subdirectories
CACHE_CR_DIR = f"{CACHE_DIR}/crs"
CACHE_SPEC_DIR = f"{CACHE_DIR}/specs"
CACHE_INDEX_DIR = f"{CACHE_DIR}/indices"
CACHE_CHUNK_DIR = f"{CACHE_DIR}/chunks"
CACHE_INSTANCE_DIR = f"{CACHE_DIR}/instances"
CACHE_IMAGE_DIR = f"{CACHE_DIR}/images"
CONV_CACHE_DIR = f"{CACHE_DIR}/conversations"
BATCH_INPUT_DIR = f"{CACHE_DIR}/batch_inputs"
BATCH_OUTPUT_DIR = f"{CACHE_DIR}/batch_outputs"
CACHE_DATASET_DIR = f"{CACHE_DIR}/datasets"
CACHE_DATASET_MESSAGES_DIR = f"{CACHE_DATASET_DIR}/messages"
CACHE_DATASET_TOKENIZED_DIR = f"{CACHE_DATASET_DIR}/conversations"

MULTI_CARD_COMMUNICATION_CACHE_DIR = f"{CACHE_DIR}/multi_card_communication"

# Log subdirectories
LOG_VLLM_SERVER_DIR = f"{LOG_DIR}/vllm_servers"

SYSTEM_INST_WORDING_DIR = f"{ASSET_DIR}/system_instructions"


# Function to create directories if they do not exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# Create all necessary directories
DIRECTORIES = [
    CACHE_DIR,
    LOG_DIR,
    CACHE_CR_DIR,
    CACHE_SPEC_DIR,
    CACHE_INDEX_DIR,
    ASSET_DIR,
    CACHE_CHUNK_DIR,
    CACHE_INSTANCE_DIR,
    CACHE_IMAGE_DIR,
    CONV_CACHE_DIR,
    BATCH_INPUT_DIR,
    BATCH_OUTPUT_DIR,
    LOG_VLLM_SERVER_DIR,
    CACHE_DATASET_DIR,
    CACHE_DATASET_MESSAGES_DIR,
    CACHE_DATASET_TOKENIZED_DIR,
    SYSTEM_INST_WORDING_DIR,
    MULTI_CARD_COMMUNICATION_CACHE_DIR,
]

for dir in DIRECTORIES:
    create_directory(dir)
