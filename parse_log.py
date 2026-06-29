"""
Text Training Log Parser and Reward Visualizer

This script parses training rewards from a standard text log file (e.g., log_train.txt)
using regular expressions, computes a moving average to smooth the reward curve, 
and saves the resulting visualization as a PNG plot.

Example usage:
    python parse_log.py -p log/tvsum_split0/log_train.txt
"""

import os
import argparse
import re
import os.path as osp
import matplotlib
# Use standard 'Agg' backend to allow generating plots without a GUI environment
matplotlib.use('Agg')
from matplotlib import pyplot as plt
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
parser = argparse.ArgumentParser(description="Parse reward values from text logs and plot overall training progress.")
parser.add_argument('-p', '--path', type=str, required=True, 
                    help="Path to the training log file (.txt); the output plot will be saved in the same directory.")
args = parser.parse_args()

# Validate that the log file path exists
if not osp.exists(args.path):
    raise ValueError("Given path is invalid: {}".format(args.path))

# Ensure the log file ends with the .txt extension
if osp.splitext(osp.basename(args.path))[-1] != '.txt':
    raise ValueError("File found does not end with .txt: {}".format(args.path))

# Compile a regular expression to search for lines matching "reward <float_value>"
regex_reward = re.compile(r'reward ([\.\deE+-]+)')
rewards = []

# Read through the log file line by line to extract matching rewards
with open(args.path, 'r') as f:
    lines = f.readlines()
    for line in lines:
        reward_match = regex_reward.search(line)
        if reward_match:
            reward = float(reward_match.group(1))
            rewards.append(reward)

# Convert list to array and apply smoothing filter
rewards = np.array(rewards)
rewards = movingaverage(rewards, 8)

# Generate and customize the reward plot
plt.plot(rewards)
plt.xlabel('epoch')
plt.ylabel('reward')
plt.title("Overall rewards")

# Save the plot image in the directory of the log file
save_path = osp.join(osp.dirname(args.path), 'overall_reward.png')
plt.savefig(save_path)
plt.close()
