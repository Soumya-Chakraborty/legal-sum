import os
import json
import numpy as np
import pytest
import torch

def test_legal_courtroom_dataset(tmp_path):
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    annotations_path = tmp_path / "annotations.json"
    
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
    np.save(features_dir / "video1_textual.npy", np.random.randn(100, 768))
    
    from demo.legal_dataset import LegalCourtroomDataset
    dataset = LegalCourtroomDataset(
        annotations_path=str(annotations_path),
        features_dir=str(features_dir),
        num_classes=2,
        num_roles=2
    )
    
    assert len(dataset) == 1
    vis, ac, txt, ev, sp, imp = dataset[0]
    
    assert vis.shape == (100, 1024)
    assert ac.shape == (100, 40)
    assert txt.shape == (100, 768)
    assert ev.shape == (100, 2)
    assert sp.shape == (100,)
    assert imp.shape == (100,)
    
    assert ev[9, 0] == 0.0
    assert ev[10, 0] == 1.0
    assert ev[30, 0] == 0.0
    
    assert sp[0] == 0
    assert sp[39] == 0
    assert sp[40] == 1
    assert sp[99] == 1
    
    assert imp[9] == 0.0
    assert imp[10] == 0.9
    assert imp[40] == 0.8

    # Key-based lookup test
    item = dataset["video1"]
    assert isinstance(item, dict)
    assert "features" in item
    assert "acoustic" in item
    assert "semantic" in item
    assert "change_points" in item
    assert "n_frames" in item
    assert "n_frame_per_seg" in item
    assert "picks" in item
    assert "user_summary" in item
    assert "gtscore" in item

    assert item["n_frames"][()] == 100
    assert item["change_points"].shape == (20, 2)
    assert item["n_frame_per_seg"].tolist() == [5] * 20
    assert item["picks"].tolist() == list(range(100))
    assert item["user_summary"].shape == (1, 100)
