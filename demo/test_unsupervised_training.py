import os
import json
import subprocess
import numpy as np
import pytest

def test_unsupervised_courtroom_training(tmp_path):
    # 1. Setup mock features directory and files
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    annotations_path = tmp_path / "annotations.json"
    splits_path = tmp_path / "splits.json"
    
    # Create mock annotation
    annotations = {
        "video1": {
            "n_frames": 100,
            "fps": 10.0,
            "events": [
                {"type": "objection", "start_time": 1.0, "end_time": 3.0, "label_id": 0},
                {"type": "ruling", "start_time": 4.0, "end_time": 6.0, "label_id": 1}
            ],
            "speakers": [
                {"role": "lawyer", "start_time": 0.0, "end_time": 4.0, "label_id": 0},
                {"role": "judge", "start_time": 4.0, "end_time": 10.0, "label_id": 1}
            ],
            "importance": [
                {"start_time": 1.0, "end_time": 3.0, "score": 0.9},
                {"start_time": 4.0, "end_time": 6.0, "score": 0.8}
            ]
        }
    }
    
    with open(annotations_path, "w") as f:
        json.dump(annotations, f)
        
    np.save(features_dir / "video1_visual.npy", np.random.randn(100, 1024))
    np.save(features_dir / "video1_acoustic.npy", np.random.randn(100, 40))
    np.save(features_dir / "video1_textual.npy", np.random.randn(100, 512)) # 512-dim matching models.py proj_s
    
    # Create splits file
    splits = [
        {
            "train_keys": ["video1"],
            "test_keys": ["video1"]
        }
    ]
    with open(splits_path, "w") as f:
        json.dump(splits, f)
        
    # 2. Run main.py using subprocess to verify it trains for 1 epoch
    cmd = [
        "python3", "main.py",
        "-d", "dummy",  # Required but ignored for courtroom dataset-type
        "-s", str(splits_path),
        "-m", "tvsum",
        "--dataset-type", "courtroom",
        "--annotations", str(annotations_path),
        "--features-dir", str(features_dir),
        "--input-dim", "1024",
        "--hidden-dim", "256",
        "--model-type", "enhanced",
        "--max-epoch", "1",
        "--phase2-epochs", "0",  # Skip phase 2 for quick verification
        "--num-episode", "2",    # Small number of episodes
        "--save-dir", str(tmp_path / "log"),
        "--use-cpu"
    ]
    
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    print("STDOUT:", result.stdout)
    print("STDERR:", result.stderr)
    
    # Assert successful exit code
    assert result.returncode == 0, f"Training failed with stderr: {result.stderr}"
    
    # Assert that training logs were printed
    assert "epoch 1/1" in result.stdout or "epoch 1/1" in result.stderr
