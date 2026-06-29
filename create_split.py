"""
Dataset Split Generator for Video Summarization

This script generates random train/test splits for video summarization models
from an H5 dataset. The generated splits are saved as a JSON file, which contains
a list of dictionary objects representing train and test keys.

Example usage:
    python create_split.py -d datasets/tvsum.h5 --num-splits 5 --train-percent 0.8
"""

from __future__ import print_function
import os
import os.path as osp
import argparse
import h5py
import math
import numpy as np

from utils import write_json

# Parse command line arguments
parser = argparse.ArgumentParser("Code to create splits in json form")
parser.add_argument('-d', '--dataset', type=str, required=True, 
                    help="Path to the h5 dataset file containing video features and ground truth.")
parser.add_argument('--save-dir', type=str, default='datasets', 
                    help="Directory to save the generated JSON file (default: 'datasets/')")
parser.add_argument('--save-name', type=str, default='splits', 
                    help="Base filename of the saved JSON file, excluding extension (default: 'splits')")
parser.add_argument('--num-splits', type=int, default=5, 
                    help="Number of random train/test splits to generate (default: 5)")
parser.add_argument('--train-percent', type=float, default=0.8, 
                    help="Proportion of the dataset to allocate for training (default: 0.8)")

args = parser.parse_args()

def split_random(keys, num_videos, num_train):
    """
    Randomly splits video keys into training and testing sets.

    Args:
        keys (list): List of all video keys/IDs in the dataset.
        num_videos (int): Total number of videos in the dataset.
        num_train (int): Number of videos to assign to the training set.

    Returns:
        tuple: (train_keys, test_keys) where train_keys and test_keys are
               non-overlapping lists of video IDs.
    """
    train_keys, test_keys = [], []
    # Randomly select indices for the training set without replacement
    rnd_idxs = np.random.choice(range(num_videos), size=num_train, replace=False)
    for key_idx, key in enumerate(keys):
        if key_idx in rnd_idxs:
            train_keys.append(key)
        else:
            test_keys.append(key)

    # Sanity check to ensure there is no overlap between training and testing keys
    assert len(set(train_keys) & set(test_keys)) == 0, "Error: train_keys and test_keys overlap"

    return train_keys, test_keys

def create():
    """
    Main function to load the H5 dataset, compute training/testing split sizes,
    generate the specified number of random splits, and save the result as a JSON file.
    """
    print("==========\nArgs:{}\n==========".format(args))
    print("Goal: randomly split data for {} times, {:.1%} for training and the rest for testing".format(args.num_splits, args.train_percent))
    print("Loading dataset from {}".format(args.dataset))
    
    # Load dataset keys representing video identifiers
    dataset = h5py.File(args.dataset, 'r')
    keys = list(dataset.keys())
    num_videos = len(keys)
    
    # Calculate the training/testing sizes
    num_train = int(math.ceil(num_videos * args.train_percent))
    num_test = num_videos - num_train
    print("Split breakdown: # total videos {}. # train videos {}. # test videos {}".format(num_videos, num_train, num_test))
    
    splits = []
    # Generate splits
    for split_idx in range(args.num_splits):
        train_keys, test_keys = split_random(keys, num_videos, num_train)
        splits.append({
            'train_keys': train_keys,
            'test_keys': test_keys,
            })

    # Create directories if they do not exist
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    saveto = osp.join(args.save_dir, args.save_name + '.json')
    write_json(splits, saveto)
    print("Splits saved to {}".format(saveto))

    dataset.close()

if __name__ == '__main__':
    create()