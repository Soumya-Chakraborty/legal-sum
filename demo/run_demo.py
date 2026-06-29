import os
import sys
import time
import h5py
import numpy as np
import torch

# Add parent directory to path so we can import models and utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import DSN
import vsum_tools
from utils import read_json

def run_demo():
    print("==========================================================")
    print("        SCSL-SGC Unsupervised Video Summarization Demo     ")
    print("==========================================================\n")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    dataset_path = os.path.join(base_dir, 'datasets/eccv16_dataset_summe_google_pool5.h5')
    split_path = os.path.join(base_dir, 'datasets/summe_splits.json')
    checkpoint_path = os.path.join(base_dir, 'log/summe-counterfactual-optimized/model_best.pth.tar')

    # Check file existence
    for path in [dataset_path, split_path, checkpoint_path]:
        if not os.path.exists(path):
            print(f"Error: Required file not found: {path}")
            return

    # Load test split keys
    splits = read_json(split_path)
    # Split 1 test keys
    test_keys = splits[1]['test_keys']
    demo_video = test_keys[0]  # Let's run on the first test video

    # Load model
    print("Initializing SCSL-SGC Model...")
    model = DSN(in_dim=1024, hid_dim=256, num_layers=2, cell='lstm', num_heads=8, dropout=0.40)
    print(f"Loading SOTA checkpoint from: {checkpoint_path}")
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    # Load video data
    print(f"Loading video features for: {demo_video}")
    with h5py.File(dataset_path, 'r') as dataset:
        seq = dataset[demo_video]['features'][...]
        cps = dataset[demo_video]['change_points'][...]
        num_frames = dataset[demo_video]['n_frames'][()]
        nfps = dataset[demo_video]['n_frame_per_seg'][...].tolist()
        positions = dataset[demo_video]['picks'][...]
        user_summary = dataset[demo_video]['user_summary'][...]

        # Perform inference
        seq_tensor = torch.from_numpy(seq).unsqueeze(0)
        
        start_time = time.time()
        with torch.no_grad():
            probs = model(seq_tensor).squeeze().numpy()
        inference_time = time.time() - start_time

        # Generate binary summary
        machine_summary = vsum_tools.generate_summary(probs, cps, num_frames, nfps, positions)
        
        # Evaluate
        fm, prec, rec = vsum_tools.evaluate_summary(machine_summary, user_summary, eval_metric='max')

    # Convert binary frame-level summary to intervals
    selected_frames = np.where(machine_summary == 1)[0]
    intervals = []
    if len(selected_frames) > 0:
        start = selected_frames[0]
        for i in range(1, len(selected_frames)):
            if selected_frames[i] != selected_frames[i-1] + 1:
                intervals.append((start, selected_frames[i-1]))
                start = selected_frames[i]
        intervals.append((start, selected_frames[-1]))

    print("\n---------------- DEMO EVALUATION RESULTS ----------------")
    print(f"Video Name:                 {demo_video}")
    print(f"Total Original Frames:      {num_frames} frames (~{num_frames/30:.1f} seconds)")
    print(f"Selected Summary Frames:    {len(selected_frames)} frames (~{len(selected_frames)/30:.1f} seconds)")
    print(f"Inference Time:             {inference_time*1000:.2f} ms")
    print(f"F1-Score:                   {fm:.1%}")
    print(f"Precision:                  {prec:.1%}")
    print(f"Recall:                     {rec:.1%}")
    print("---------------------------------------------------------")

    print("\nSelected Video Summarization Intervals:")
    for idx, (s, e) in enumerate(intervals):
        print(f"  Shot {idx+1:02d}: Frames {s:05d} to {e:05d} (Duration: {(e-s)/30:.2f}s)")
    print("==========================================================")

if __name__ == '__main__':
    run_demo()
