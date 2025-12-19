import os
import ast
import torch
import socket
import hashlib
import argparse


def is_port_open(host, port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        return sock.connect_ex((host, port)) == 0


def get_project_base_directory():
    return os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )


def get_project_directory(*args):
    return os.path.join(get_project_base_directory(), *args)


def get_home_cache_dir():
    dir = os.path.join(os.path.expanduser("~"), ".critic")
    try:
        os.mkdir(dir)
    except OSError as error:
        pass
    return dir


def generate_16_char_hash(input_string):
    # Create an MD5 hash object
    hash_object = hashlib.md5()

    # Encode the input string to bytes and update the hash object
    hash_object.update(input_string.encode("utf-8"))

    # Get the hexadecimal representation of the hash
    full_hash = hash_object.hexdigest()

    # Truncate the hash to the first 16 characters
    truncated_hash = full_hash[:16]

    return truncated_hash


def parse_dict(s):
    if s is None:
        return None

    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):

        raise argparse.ArgumentTypeError("Input should be a valid dictionary string.")


def print_rank_0(*message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*message, flush=True)
    else:
        print(*message, flush=True)


def warning_rank_0(*message):
    BAR = "*" * 80
    print_rank_0(BAR)
    print_rank_0(*message)
    print_rank_0(BAR)
