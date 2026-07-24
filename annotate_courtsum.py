"""
annotate_courtsum.py — Human/Legal Expert Annotation Pipeline for COURTSUM

1. --export: Exports CSV annotation sheets for each trial video in COURTSUM.
   Lists 5-second video segments with timestamps (mm:ss) so legal experts can rate
   segment importance from 1 (irrelevant) to 5 (critical legal event).

2. --import: Imports completed CSV files from experts, calculates true gtscore (mean 0-1 ratings),
   constructs the user_summary matrix (N_experts x T binary selections via 15% Knapsack budget),
   and updates datasets/eccv16_dataset_courtsum_google_pool5.h5.
"""

import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
import h5py
from knapsack import knapsack_dp

H5_PATH = "datasets/eccv16_dataset_courtsum_google_pool5.h5"
TEMPLATES_DIR = "annotations_expert/templates"
RESPONSES_DIR = "annotations_expert/responses"


def format_timestamp(seconds: float) -> str:
    """Format seconds into HH:MM:SS."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def export_templates(blank_for_crowdsource=False):
    """Export CSV annotation sheets for legal experts / crowdsourcing."""
    if not os.path.exists(H5_PATH):
        print(f"[Error] Dataset {H5_PATH} not found. Run build_courtsum.py first.")
        return

    target_dir = "annotations_expert/crowdsourcing_templates" if blank_for_crowdsource else TEMPLATES_DIR
    os.makedirs(target_dir, exist_ok=True)
    os.makedirs(RESPONSES_DIR, exist_ok=True)

    with h5py.File(H5_PATH, "r") as h5f:
        for video_key in h5f.keys():
            grp = h5f[video_key]
            n_frames = int(grp["n_frames"][()])
            fps = float(grp["fps"][()]) if "fps" in grp else 2.0
            change_points = grp["change_points"][...]

            rows = []
            for seg_idx, (start_f, end_f) in enumerate(change_points):
                start_sec = start_f / fps
                end_sec = (end_f + 1) / fps
                rows.append({
                    "segment_id": seg_idx + 1,
                    "timestamp_start": format_timestamp(start_sec),
                    "timestamp_end": format_timestamp(end_sec),
                    "frame_start": int(start_f),
                    "frame_end": int(end_f),
                    "rating_1_to_5": "" if blank_for_crowdsource else 3,
                    "trial_phase": "" if blank_for_crowdsource else "Witness Examination",
                    "annotator_id": "",
                    "notes": ""
                })

            df = pd.DataFrame(rows)
            suffix = "_crowdsource_blank.csv" if blank_for_crowdsource else "_expert_template.csv"
            csv_path = os.path.join(target_dir, f"{video_key}{suffix}")
            df.to_csv(csv_path, index=False)
            print(f"[Exported Template]: {csv_path} ({len(rows)} segments)")

    print(f"\n[Success] Blank templates exported to: {target_dir}/")
    print("Instruct legal experts to rate segment importance (1-5) and save completed CSVs in:")
    print(f"  {RESPONSES_DIR}/ (e.g. {video_key}_expert1.csv, {video_key}_expert2.csv)")


def import_annotations():
    """Import expert CSV ratings and update HDF5 dataset."""
    if not os.path.exists(H5_PATH):
        print(f"[Error] Dataset {H5_PATH} not found.")
        return

    response_files = glob.glob(os.path.join(RESPONSES_DIR, "*.csv"))
    if not response_files:
        print(f"[Warning] No completed CSV responses found in {RESPONSES_DIR}/")
        print("Please place completed expert CSV files in that folder.")
        return

    print(f"Found {len(response_files)} expert response files.")

    # Group response files by video_key
    video_responses = {}
    with h5py.File(H5_PATH, "r") as h5f:
        valid_keys = list(h5f.keys())

    for fpath in response_files:
        fname = os.path.basename(fpath)
        for vkey in valid_keys:
            if fname.startswith(vkey):
                if vkey not in video_responses:
                    video_responses[vkey] = []
                video_responses[vkey].append(fpath)
                break

    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    with h5py.File(H5_PATH, "r+") as h5f:
        for vkey, fpaths in video_responses.items():
            grp = h5f[vkey]
            n_frames = int(grp["n_frames"][()])
            change_points = grp["change_points"][...]
            num_segments = len(change_points)

            expert_ratings = []
            for fp in fpaths:
                df = pd.read_csv(fp)
                if "rating_1_to_5" in df.columns:
                    ratings = df["rating_1_to_5"].values
                    # Clip ratings to 1..5
                    ratings = np.clip(ratings, 1, 5)
                    expert_ratings.append(ratings)

            if not expert_ratings:
                continue

            expert_ratings = np.array(expert_ratings, dtype=np.float32)  # (N_experts, num_segments)
            n_experts = len(expert_ratings)

            # 1. Compute per-segment mean rating (1..5) -> scale to [0, 1]
            seg_mean_ratings = expert_ratings.mean(axis=0)  # (num_segments,)
            seg_scores_norm = (seg_mean_ratings - 1.0) / 4.0  # Normalize 1..5 to 0..1

            # Expand segment scores to per-frame gtscore
            gtscore = np.zeros(n_frames, dtype=np.float32)
            for seg_idx, (sf, ef) in enumerate(change_points):
                gtscore[sf:ef+1] = seg_scores_norm[seg_idx]

            # 2. Build user_summary matrix (N_experts x n_frames) using 15% Knapsack budget
            user_summary = np.zeros((n_experts, n_frames), dtype=np.int32)
            target_budget = int(np.ceil(0.15 * n_frames))

            for exp_idx in range(n_experts):
                exp_seg_scores = (expert_ratings[exp_idx] - 1.0) / 4.0
                exp_frame_scores = np.zeros(n_frames, dtype=np.float32)
                for seg_idx, (sf, ef) in enumerate(change_points):
                    exp_frame_scores[sf:ef+1] = exp_seg_scores[seg_idx]

                # Knapsack selection per segment
                weights = [int(ef - sf + 1) for sf, ef in change_points]
                values = [float(exp_seg_scores[i] * weights[i]) for i in range(num_segments)]
                selected_segs = knapsack_dp(values, weights, len(values), target_budget)

                for s_idx in selected_segs:
                    sf, ef = change_points[s_idx]
                    user_summary[exp_idx, sf:ef+1] = 1

            # Update HDF5 dataset in-place
            if "gtscore" in grp:
                del grp["gtscore"]
            if "user_summary" in grp:
                del grp["user_summary"]

            grp.create_dataset("gtscore", data=gtscore)
            grp.create_dataset("user_summary", data=user_summary)

            print(f"[Updated Dataset {vkey}]: {n_experts} experts imported.")
            print(f"  gtscore mean: {gtscore.mean():.3f}, user_summary shape: {user_summary.shape}")

    print(f"\n[Complete] HDF5 dataset {H5_PATH} successfully updated with expert annotations!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Human/Legal Expert Annotation Pipeline for COURTSUM")
    parser.add_argument("--export", action="store_true", help="Export CSV annotation templates for experts")
    parser.add_argument("--blank", action="store_true", help="Export completely BLANK CSV templates for crowdsourcing")
    parser.add_argument("--import-responses", action="store_true", help="Import completed expert CSVs and update HDF5")
    args = parser.parse_args()

    if args.blank:
        export_templates(blank_for_crowdsource=True)
    elif args.export:
        export_templates(blank_for_crowdsource=False)
    elif args.import_responses:
        import_annotations()
    else:
        export_templates(blank_for_crowdsource=True)
