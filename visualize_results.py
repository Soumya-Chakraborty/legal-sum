"""
Video Summarization Results Visualizer

This script reads video summarization results from an H5 file, generates
plots comparing ground truth importance scores with predicted scores, and
saves the plots as PNG images.

Example usage:
    python visualize_results.py -p path/to/results.h5
"""

import h5py
from matplotlib import pyplot as plt
import argparse
import os
import os.path as osp

# Set up argument parser to specify the path to the H5 results file
parser = argparse.ArgumentParser(description="Visualize video summarization results from H5 files.")
parser.add_argument('-p', '--path', type=str, required=True,
                    help="path to h5 file containing summarization results")
args = parser.parse_args()

# Open the H5 file containing the evaluation results in read-only mode
h5_res = h5py.File(args.path, 'r')
keys = list(h5_res.keys())  # Each key represents a unique video ID

# Iterate through all videos stored in the H5 file
for key in keys:
    # Extract prediction and ground-truth values:
    # - score: Predicted frame-level importance scores
    # - machine_summary: Binary summary array (1 for selected frames, 0 otherwise)
    # - gtscore: Ground truth frame-level importance scores
    # - fm: The F1-score achieved by the model on this video
    score = h5_res[key]['score'][...]
    machine_summary = h5_res[key]['machine_summary'][...]
    gtscore = h5_res[key]['gtscore'][...]
    fm = h5_res[key]['fm'][()]

    # Set up a plot with two subplots:
    # axs[0] for ground truth score, axs[1] for predicted score
    fig, axs = plt.subplots(2)
    n = len(gtscore)  # Number of frames in the video
    
    # Plot ground-truth scores on the top subplot (in red)
    axs[0].plot(range(n), gtscore, color='red')
    axs[0].set_xlim(0, n)
    axs[0].set_yticklabels([])  # Hide y-axis labels for a cleaner look
    axs[0].set_xticklabels([])  # Hide x-axis labels

    # Plot predicted scores on the bottom subplot (in blue)
    axs[1].set_title("video {} F-score {:.1%}".format(key, fm))
    axs[1].plot(range(n), score, color='blue')
    axs[1].set_xlim(0, n)
    axs[1].set_yticklabels([])  # Hide y-axis labels
    axs[1].set_xticklabels([])  # Hide x-axis labels

    # Save the generated figure into the same directory as the H5 file
    save_path = osp.join(osp.dirname(args.path), 'score_' + key + '.png')
    fig.savefig(save_path, bbox_inches='tight')
    plt.close()  # Close the plot to free memory

    # Print confirmation message for this video (Python 3 function syntax)
    print("Done video {}. # frames {}.".format(key, len(machine_summary)))

# Close the H5 file resource properly
h5_res.close()