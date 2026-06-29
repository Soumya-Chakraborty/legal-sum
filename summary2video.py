"""
Video Summarization Frame-to-Video Compiler

This script reads the binary video summarization results (machine summaries)
from an H5 file, retrieves corresponding raw frame images from a directory,
filters/selects only the summary frames, and compiles them into a summary video (.mp4).

Example usage:
    python summary2video.py -p log/results.h5 -d data/frames/video_1/ -i 0 --save-dir outputs/
"""

import h5py
import cv2
import os
import os.path as osp
import numpy as np
import argparse

# Configure argument parser
parser = argparse.ArgumentParser(description="Compile summarized frame sequences back into a video file.")
parser.add_argument('-p', '--path', type=str, required=True, 
                    help="Path to the H5 results file containing machine_summary.")
parser.add_argument('-d', '--frm-dir', type=str, required=True, 
                    help="Directory containing the raw video frame images (e.g., JPEG files).")
parser.add_argument('-i', '--idx', type=int, default=0, 
                    help="Index of the video key to extract summary frames from (0-based, default: 0).")
parser.add_argument('--fps', type=int, default=30, 
                    help="Frame rate (FPS) of the output compiled video (default: 30).")
parser.add_argument('--width', type=int, default=640, 
                    help="Target frame width for the output video (default: 640).")
parser.add_argument('--height', type=int, default=480, 
                    help="Target frame height for the output video (default: 480).")
parser.add_argument('--save-dir', type=str, default='log', 
                    help="Directory where the output video will be saved (default: 'log').")
parser.add_argument('--save-name', type=str, default='summary.mp4', 
                    help="Output video filename, ending with .mp4 (default: 'summary.mp4').")
args = parser.parse_args()


def frm2video(frm_dir, summary, vid_writer):
    """
    Iterates through the frame summary array, loads selected frame images, 
    resizes them, and writes them to the target video using OpenCV.

    Args:
        frm_dir (str): Directory containing the sequential JPEG frames.
        summary (ndarray): Binary summary array where 1 means keep the frame and 0 means exclude.
        vid_writer (cv2.VideoWriter): OpenCV video writer object initialized with output settings.
    """
    for idx, val in enumerate(summary):
        # Only write frames that are selected by the machine summary
        if val == 1:
            # Assumes 1-based indexing for frame filenames (e.g., '000001.jpg')
            # Padded to 6 characters. Change zfill value or format if necessary.
            frm_name = str(idx + 1).zfill(6) + '.jpg'
            frm_path = osp.join(frm_dir, frm_name)
            
            frm = cv2.imread(frm_path)
            if frm is not None:
                frm = cv2.resize(frm, (args.width, args.height))
                vid_writer.write(frm)
            else:
                print("Warning: Frame not found at path: {}".format(frm_path))


if __name__ == '__main__':
    # Ensure the output directory exists
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Initialize the OpenCV VideoWriter object with the MP4V codec
    output_path = osp.join(args.save_dir, args.save_name)
    vid_writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'MP4V'),
        args.fps,
        (args.width, args.height),
    )

    # Read the summary indices from the results file
    h5_res = h5py.File(args.path, 'r')
    # Cast keys to list to make it indexable under Python 3
    keys = list(h5_res.keys())
    assert args.idx < len(keys), "Error: Provided index {} exceeds number of video keys ({}).".format(args.idx, len(keys))
    key = keys[args.idx]
    summary = h5_res[key]['machine_summary'][...]
    h5_res.close()

    # Compile the frames into the summary video
    frm2video(args.frm_dir, summary, vid_writer)
    
    # Release the video writer to save and finalize the video file
    vid_writer.release()
    print("Summary video saved successfully at: {}".format(output_path))