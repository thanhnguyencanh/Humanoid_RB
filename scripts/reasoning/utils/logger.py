from time import time
import logging
import torch
from colorlog import ColoredFormatter
import os
import cv2
import json
import numpy as np

def sync_time():
    torch.cuda.synchronize()
    return time()


Log = logging.getLogger()
Log.time = time
Log.sync_time = sync_time

# Set default
Log.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
# Use colorlog
formatstring = "[%(cyan)s%(asctime)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] %(message)s"
datefmt = "%m/%d %H:%M:%S"
ch.setFormatter(ColoredFormatter(formatstring, datefmt=datefmt))

Log.addHandler(ch)
# Log.info("Init-Logger")


def timer(sync_cuda=False, mem=False, loop=1):
    """
    Args:
        func: function
        sync_cuda: bool, whether to synchronize cuda
        mem: bool, whether to log memory
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if mem:
                start_mem = torch.cuda.memory_allocated() / 1024**2
            if sync_cuda:
                torch.cuda.synchronize()

            start = Log.time()
            for _ in range(loop):
                result = func(*args, **kwargs)

            if sync_cuda:
                torch.cuda.synchronize()
            if loop == 1:
                message = f"{func.__name__} took {Log.time() - start:.3f} s."
            else:
                message = f"{func.__name__} took {((Log.time() - start))/loop:.3f} s. (loop={loop})"

            if mem:
                end_mem = torch.cuda.memory_allocated() / 1024**2
                end_max_mem = torch.cuda.max_memory_allocated() / 1024**2
                message += f" Start_Mem {start_mem:.1f} Max {end_max_mem:.1f} MB"
            Log.info(message)

            return result
        return wrapper
    return decorator

def save_to_path(path, data):
    """
    Saves 'data' to 'path'.
    Automatically creates necessary folders and detects data type.
    """
    # 1. Create the folder structure if it doesn't exist
    folder_dir = os.path.dirname(path)
    if folder_dir and not os.path.exists(folder_dir):
        os.makedirs(folder_dir, exist_ok=True)

    # 2. Save based on data type
    try:
        if isinstance(data, np.ndarray):
            # It's an image (OpenCV format)
            cv2.imwrite(path, data)

        elif isinstance(data, (dict, list)):
            # It's metadata/JSON
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

        elif isinstance(data, bytes):
            # It's binary data
            with open(path, 'wb') as f:
                f.write(data)

        else:
            # It's a string or other text
            with open(path, 'w', encoding='utf-8') as f:
                f.write(str(data))

        # print(f"Saved: {path}") # Optional logging

    except Exception as e:
        print(f"Error saving to {path}: {e}")

def timed(fn):
    """example usage: timed(lambda: model(inp))"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000