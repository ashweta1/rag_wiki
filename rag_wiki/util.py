"""
This file contains utility functions that are used in the RAG Wiki library.
"""
import torch

override_device = None

def best_device():
    if override_device:
        device = override_device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    return device

def is_gpu(device):
    return device == "cuda" or device == "mps"

def print_debug(debug, message):
    if debug:
        print(message)