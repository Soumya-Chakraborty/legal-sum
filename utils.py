"""
Utility Functions and Classes for Video Summarization

This module provides common helper utilities, including directory creation,
metrics tracking (AverageMeter), file checkpointing, redirection of standard
output to log files (Logger), and JSON file reading/writing.
"""

from __future__ import absolute_import
import os
import sys
import errno
import shutil
import json
import os.path as osp

import torch


def mkdir_if_missing(directory):
    """
    Creates a directory if it does not already exist. Handles potential race
    conditions where directory creation could collide.

    Args:
        directory (str): Path of the directory to be created.
    """
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            # Silence error if directory was created by another process/thread in parallel
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """
    Computes and stores the running average, sum, count, and current value of a metric.
    
    Used to track training/validation statistics such as loss or reward over epochs.
    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all metrics to zero."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Updates the running statistics with a new value.

        Args:
            val (float/int): The new metric value.
            n (int, optional): The frequency/weight of the new value. Defaults to 1.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, fpath='checkpoint.pth.tar'):
    """
    Saves a training checkpoint state dictionary to a file using PyTorch.

    Args:
        state (dict): Dictionary containing model weights, optimizer state, and current epoch.
        fpath (str, optional): Target file path. Defaults to 'checkpoint.pth.tar'.
    """
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)


class Logger(object):
    """
    Custom Logger class to redirect standard print/console output to an external log file
    while preserving console outputs.
    
    Adapted from: https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py
    """
    def __init__(self, fpath=None):
        self.console = sys.stdout  # Save reference to original standard output
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        # Support context manager usage
        return self

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        """Writes message to both console and the file (if open)."""
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        """Flushes buffered stream to both console and log file."""
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            # Force writing buffer contents to disk
            os.fsync(self.file.fileno())

    def close(self):
        """Closes the log file resource. Does not close stdout to avoid breaking python output."""
        if self.file is not None:
            self.file.close()
            self.file = None


def read_json(fpath):
    """
    Utility function to read and parse a JSON file.

    Args:
        fpath (str): Path to the target JSON file.

    Returns:
        dict/list: Parsed JSON object.
    """
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    """
    Utility function to write a Python object to a JSON file.

    Args:
        obj (dict/list): Object to serialize.
        fpath (str): Target destination file path.
    """
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))






