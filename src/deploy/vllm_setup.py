import os
import time
import torch
import psutil
import signal
import atexit
import subprocess
from datetime import datetime
from src.utils.logging import logging
from src.deploy.models import model_name_conversion
from src.configs import LOG_VLLM_SERVER_DIR, LOCAL_API_KEY, HUGGINGFACE_HUB_CACHE

logger = logging.getLogger(__name__)

model_name_to_local_path = {
    "ZhipuAI/glm-4-9b": f"{HUGGINGFACE_HUB_CACHE}/ZhipuAI/glm-4-9b",
    "ZhipuAI/glm-4-9b-chat": f"{HUGGINGFACE_HUB_CACHE}/ZhipuAI/glm-4-9b-chat",
}

processes = []
server_config = {}  # Global variable to store server configuration


def signal_handler(signum, frame):
    logging.info(f"Received signal {signum}, terminating child processes...")
    stop_model_servers(processes)


def kill_gpu_processes():
    logger.info("Terminating all processes using GPU...")
    result = subprocess.run(["ps", "aux"], stdout=subprocess.PIPE)
    processes = result.stdout.decode("utf-8").split("\n")
    for process in processes:
        if "vllm.entrypoints.openai.api_server" in process and "python" in process:
            pid = int(process.split()[1])
            logger.info(f"Killing process with PID: {pid}")
            os.kill(pid, signal.SIGTERM)


def start_server(model_name, using_lora, tensor_parallel_size, cuda_device, port):
    # Create a timestamped log file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if os.path.exists(model_name):
        log_filename = f"{timestamp}_{model_name.split('/')[-1]}_{port}.log"
    else:
        log_filename = f"{timestamp}_{model_name}_{port}.log"

    log_filepath = os.path.join(LOG_VLLM_SERVER_DIR, log_filename)

    # Open the log file
    log_file = open(log_filepath, "w")

    model_name_x = (
        model_name
        if model_name not in model_name_conversion
        else model_name_conversion[model_name]
    )

    if model_name_x in model_name_to_local_path:
        model_name_or_path = model_name_to_local_path[model_name_x]
    else:
        model_name_or_path = model_name_x
        if os.path.exists(model_name_x):
            model_name_x = "local-model"

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device

    if using_lora is None or not os.path.exists(using_lora):
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name_or_path,
            "--served-model-name",
            model_name_x,
            "--dtype",
            "auto",
            "--api-key",
            LOCAL_API_KEY,
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--trust-remote-code",
            "--enforce_eager",
            "--disable-log-requests",
            "--gpu-memory-utilization",
            "0.95",
            "--enable-prefix-caching",
            "--port",
            str(port),
        ]
    else:
        cmd = [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model_name_or_path,
            "--dtype",
            "auto",
            "--api-key",
            LOCAL_API_KEY,
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--trust-remote-code",
            "--enforce_eager",
            "--gpu-memory-utilization",
            "0.95",
            "--enable-prefix-caching",
            "--port",
            str(port),
            "--guided-decoding-backend",
            "lm-format-enforcer",
            "--enable-lora",
            "--lora-modules",
            f"{model_name_x}={using_lora}",
            "--max_num_batched_tokens",
            "60000",
            "--max_model_len",
            "11000",
        ]

    if "phi" in model_name_or_path.lower():
        cmd.remove("--enable-prefix-caching")
        cmd.append("--enable-chunked-prefill=False")
        cmd.append("--max-num-seqs=20")

    logger.info(
        f"Starting server with command: {' '.join(cmd)} on CUDA device(s): {cuda_device} at port: {port}"
    )
    return subprocess.Popen(cmd, env=env, stdout=log_file, stderr=log_file), log_file


def start_model_servers(model_name, using_lora, tensor_parallel_size=1):
    global processes, server_config
    os.environ["VLLM_USE_MODELSCOPE"] = "True"

    # Get the number of available GPUs
    num_gpus = 2

    # Terminate existing GPU processes
    # kill_gpu_processes()

    processes = []

    if tensor_parallel_size > num_gpus:
        raise ValueError(
            "The tensor_parallel_size exceeds the number of available GPUs."
        )

    # Calculate the number of full groups of GPUs
    server_num = num_gpus // tensor_parallel_size

    # Create groups of CUDA devices according to tensor_parallel_size
    device_groups = [
        ",".join(map(str, range(i, i + tensor_parallel_size)))
        for i in range(0, server_num * tensor_parallel_size, tensor_parallel_size)
    ]

    ports = [8000 + i for i in range(server_num)]

    server_config = {
        "model_name": model_name,
        "tensor_parallel_size": tensor_parallel_size,
        "device_groups": device_groups,
        "ports": ports,
    }

    for cuda_devices, port in zip(device_groups, ports):
        process, log_file = start_server(
            model_name, using_lora, tensor_parallel_size, cuda_devices, port
        )
        processes.append((process, log_file))
        time.sleep(1)

    logging.info(f"Started {server_num} server(s).")
    return processes


def stop_model_servers(processes, max_retries=3, timeout=5):
    """
    Stops model server processes with retries and timeout to handle edge cases where
    processes do not terminate properly.

    Args:
        processes (list): List of tuples containing process and log file.
        max_retries (int): Maximum number of retries to attempt killing a process.
        timeout (int): Timeout in seconds to wait between retries.
    """
    for process, log_file in processes:
        try:
            if psutil.pid_exists(process.pid):
                retries = 0
                while retries < max_retries:
                    # Attempt to terminate the process
                    os.kill(process.pid, signal.SIGTERM)
                    logging.info(f"Sent SIGTERM to server with PID: {process.pid}")

                    # Wait for a short period to see if the process terminates
                    for _ in range(timeout):
                        if not psutil.pid_exists(process.pid):
                            logging.info(
                                f"Process with PID {process.pid} terminated successfully."
                            )
                            break
                        time.sleep(1)

                    # Break out of the retry loop if the process is terminated
                    if not psutil.pid_exists(process.pid):
                        break

                    # Increment retries and try again
                    retries += 1
                    logging.warning(
                        f"Retry {retries} for process with PID: {process.pid}"
                    )

                # If still not terminated after retries, use SIGKILL as a last resort
                if psutil.pid_exists(process.pid):
                    os.kill(process.pid, signal.SIGKILL)
                    logging.error(
                        f"Forced stop of process with PID: {process.pid} using SIGKILL."
                    )

                process.wait()  # Ensure the process is reaped
                log_file.close()
            else:
                logging.warning(f"Process with PID: {process.pid} is not running.")
        except Exception as e:
            logging.error(
                f"Error while stopping process with PID: {process.pid}. Exception: {e}"
            )


def restart_server_on_failure():
    global processes, server_config
    logging.info("Restarting server due to failure...")
    stop_model_servers(processes)
    time.sleep(5)
    processes = start_model_servers(
        server_config["model_name"], server_config["tensor_parallel_size"]
    )


# Setup signal handlers for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# Define a cleanup function that stops the servers
def cleanup():
    stop_model_servers(processes)


# Register atexit hook to stop servers on exit
atexit.register(cleanup)


def is_process_running(pid):
    """Check if a process with the given PID is still running."""
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


# Example of a main function to demonstrate usage
if __name__ == "__main__":
    try:
        start_model_servers("ZhipuAI/glm-4-9b", tensor_parallel_size=2)
        while True:
            time.sleep(1)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        stop_model_servers(processes)
