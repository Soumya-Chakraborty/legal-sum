import os
import json
import subprocess
import numpy as np
import pytest

def test_plotting_evaluation_training(tmp_path):
    # 1. Setup mock features directory and files
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    annotations_path = tmp_path / "annotations.json"
    splits_path = tmp_path / "splits.json"
    
    # Create mock annotation
    annotations = {
        "video1": {
            "n_frames": 50,
            "fps": 10.0,
            "events": [
                {"type": "objection", "start_time": 1.0, "end_time": 3.0, "label_id": 0},
                {"type": "ruling", "start_time": 3.0, "end_time": 5.0, "label_id": 1}
            ],
            "speakers": [
                {"role": "lawyer", "start_time": 0.0, "end_time": 3.0, "label_id": 0},
                {"role": "judge", "start_time": 3.0, "end_time": 5.0, "label_id": 1}
            ],
            "importance": [
                {"start_time": 1.0, "end_time": 3.0, "score": 0.9},
                {"start_time": 3.0, "end_time": 5.0, "score": 0.8}
            ]
        }
    }
    
    with open(annotations_path, "w") as f:
        json.dump(annotations, f)
        
    np.save(features_dir / "video1_visual.npy", np.random.randn(50, 1024))
    np.save(features_dir / "video1_acoustic.npy", np.random.randn(50, 40))
    np.save(features_dir / "video1_textual.npy", np.random.randn(50, 512))
    
    # Create splits file
    splits = [
        {
            "train_keys": ["video1"],
            "test_keys": ["video1"]
        }
    ]
    with open(splits_path, "w") as f:
        json.dump(splits, f)
        
    # 2. Run main.py using subprocess to verify training plots are generated
    save_dir = tmp_path / "log"
    cmd = [
        "python3", "main.py",
        "-d", "dummy",  # Ignored for courtroom
        "-s", str(splits_path),
        "-m", "tvsum",
        "--dataset-type", "courtroom",
        "--annotations", str(annotations_path),
        "--features-dir", str(features_dir),
        "--input-dim", "1024",
        "--hidden-dim", "128",
        "--model-type", "enhanced",
        "--max-epoch", "2",
        "--phase2-epochs", "0",
        "--num-episode", "1",
        "--save-dir", str(save_dir),
        "--use-cpu",
        "--eval-courtroom"
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    assert result.returncode == 0, f"Training failed with stderr: {result.stderr}"
    
    # 3. Assert separate graph plots exist
    plots_dir = save_dir / "plots"
    assert plots_dir.exists(), "Plots directory was not created"
    assert (plots_dir / "f_score_curve.png").exists(), "f_score_curve.png not found"
    assert (plots_dir / "correlation_curve.png").exists(), "correlation_curve.png not found"
    assert (plots_dir / "courtroom_coverage_curve.png").exists(), "courtroom_coverage_curve.png not found"
    assert (plots_dir / "reward_entropy_curve.png").exists(), "reward_entropy_curve.png not found"
