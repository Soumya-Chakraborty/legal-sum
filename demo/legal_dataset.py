import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class LegalCourtroomDataset(Dataset):
    def __init__(self, annotations_path, features_dir, num_classes=3, num_roles=3):
        with open(annotations_path, 'r') as f:
            self.annotations = json.load(f)
        self.video_keys = list(self.annotations.keys())
        self.features_dir = features_dir
        self.num_classes = num_classes
        self.num_roles = num_roles

    def __len__(self):
        return len(self.video_keys)

    def __getitem__(self, idx):
        key = self.video_keys[idx]
        anno = self.annotations[key]
        n_frames = anno['n_frames']
        fps = anno['fps']

        visual = np.load(os.path.join(self.features_dir, f"{key}_visual.npy"))
        acoustic = np.load(os.path.join(self.features_dir, f"{key}_acoustic.npy"))
        textual = np.load(os.path.join(self.features_dir, f"{key}_textual.npy"))

        event_mask = np.zeros((n_frames, self.num_classes), dtype=np.float32)
        speaker_mask = np.zeros((n_frames,), dtype=np.int64)
        importance_scores = np.zeros((n_frames,), dtype=np.float32)

        for ev in anno.get('events', []):
            start_frame = int(ev['start_time'] * fps)
            end_frame = int(ev['end_time'] * fps)
            label_id = ev['label_id']
            if label_id < self.num_classes:
                event_mask[max(0, start_frame):min(n_frames, end_frame), label_id] = 1.0

        for sp in anno.get('speakers', []):
            start_frame = int(sp['start_time'] * fps)
            end_frame = int(sp['end_time'] * fps)
            label_id = sp['label_id']
            if label_id < self.num_roles:
                speaker_mask[max(0, start_frame):min(n_frames, end_frame)] = label_id

        for imp in anno.get('importance', []):
            start_frame = int(imp['start_time'] * fps)
            end_frame = int(imp['end_time'] * fps)
            score = imp['score']
            importance_scores[max(0, start_frame):min(n_frames, end_frame)] = score

        return (
            torch.from_numpy(visual).float(),
            torch.from_numpy(acoustic).float(),
            torch.from_numpy(textual).float(),
            torch.from_numpy(event_mask).float(),
            torch.from_numpy(speaker_mask).long(),
            torch.from_numpy(importance_scores).float()
        )
