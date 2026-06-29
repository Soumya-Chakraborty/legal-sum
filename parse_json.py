"""
JSON Reward Log Parser and Visualizer

This script parses training reward logs stored in a JSON file and generates
plots representing the reward trend across training epochs for a specific video.
A moving average is applied to smooth out the variance in the reward signal.

Example usage:
    python parse_json.py -p log/tvsum_split0/rewards.json -i 0
"""

import os
import argparse
import re
import os.path as osp
import matplotlib
# Use standard 'Agg' backend to allow generating plots without a GUI environment
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from utils import read_json
import numpy as np


def movingaverage(values, window):
    """
    Computes the simple moving average (SMA) of a 1D sequence using convolution.
    This is useful for smoothing highly noisy reinforcement learning reward curves.

    Args:
        values (array-like): The input list/array of values.
        window (int): The window size for the moving average.

    Returns:
        ndarray: The smoothed values (length will be len(values) - window + 1).
    """
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


# Configure argument parsing
parser = argparse.ArgumentParser(description="Parse reward logs from JSON and plot the learning curve for a video.")
parser.add_argument('-p', '--path', type=str, required=True, 
                    help="Path to the rewards.json file; the generated plot will be saved to the same directory.")
parser.add_argument('-i', '--idx', type=int, default=0, 
                    help="Index of the video key to visualize (0-based, default: 0).")
args = parser.parse_args()

# Load the JSON file containing reward logs
reward_writers = read_json(args.path)
keys = list(reward_writers.keys())

# Validate that the selected index exists in the dictionary keys
assert args.idx < len(keys), "Error: Index {} is out of range. Total keys available: {}".format(args.idx, len(keys))
key = keys[args.idx]
rewards = reward_writers[key]

# Convert the rewards to a NumPy array and smooth using a moving average window of 8
rewards = np.array(rewards)
rewards = movingaverage(rewards, 8)

# Plot the reward curve
plt.plot(rewards)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title("Reward Curve for Video: {}".format(key))

# Save the plot in the same directory as the rewards.json file
save_path = osp.join(osp.dirname(args.path), 'epoch_reward_' + str(args.idx) + '.png')
plt.savefig(save_path)
plt.close()