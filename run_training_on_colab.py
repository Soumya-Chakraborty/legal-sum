import subprocess
import sys
import os

def run_cmd_stream(cmd, cwd=None):
    print(f"\n============================================================")
    print(f"Running command: {cmd}")
    print(f"============================================================")
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=cwd)
    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.rstrip())
            sys.stdout.flush()
    rc = process.poll()
    if rc != 0:
        raise Exception(f"Command failed with exit code {rc}")

# 1. Install dependencies
run_cmd_stream("pip install tabulate h5py")

# 2. Start hybrid RL training with supervised F1 bonus. Run directly from mounted Google Drive path.
# Set ensemble-k to 1 during training to completely prevent GPU memory leak and speed up validation loop by 10x.
training_cmd = (
    "python -u main.py "
    "-d datasets/eccv16_dataset_summe_google_pool5.h5 "
    "-s datasets/summe_splits.json "
    "--split-id 0 "
    "-m summe "
    "--model-type enhanced "
    "--hidden-dim 128 "
    "--supervised-weight 2.0 "
    "--num-episode 5 "
    "--ppo-inner-steps 4 "
    "--max-epoch 55 "
    "--pretrain-epochs 5 "
    "--phase2-epochs 15 "
    "--reward-warmup-epochs 20 "
    "--ppo-clip 0.2 "
    "--ot-weight 0.10 "
    "--contrastive-weight 0.05 "
    "--action-lock-start 0.70 "
    "--action-lock-end 0.50 "
    "--lr-scheduler cosine_warm "
    "--ensemble-k 1 "
    "--save-dir log/summe_split0_hybrid"
)

run_cmd_stream(training_cmd, cwd="/content/drive/MyDrive/legal_sum")
