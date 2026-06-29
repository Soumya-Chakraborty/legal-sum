import os
import sys
import time
import json
import urllib.request
import h5py
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

# Add parent directory to path so we can import model and solver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import DSN
from knapsack import knapsack_dp
from demo.legal_sum import run_legal_sum

def main():
    print("==========================================================")
    print("      COURT PROCEEDING TRAINING & SUMMARIZATION DEMO      ")
    print("==========================================================\n")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    # 1. Download Nuremberg Trial court proceeding video
    video_url = "https://upload.wikimedia.org/wikipedia/commons/b/be/Ministries_trial_arraignment.webm"
    court_video = os.path.join(base_dir, 'demo/court_trial.webm')
    
    if not os.path.exists(court_video):
        print(f"Downloading trial video from: {video_url}")
        req = urllib.request.Request(video_url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as res, open(court_video, 'wb') as f:
            f.write(res.read())
        print("Download complete.")
    else:
        print("Trial video already downloaded.")
        
    # 2. Extract Features to custom dataset H5 file
    h5_path = os.path.join(base_dir, 'demo/court_dataset.h5')
    print(f"\nProcessing features and writing to: {h5_path}")
    
    # Load feature extractor GoogLeNet
    try:
        weights = tv_models.GoogLeNet_Weights.DEFAULT
        feature_model = tv_models.googlenet(weights=weights)
    except AttributeError:
        feature_model = tv_models.googlenet(pretrained=True)
    feature_model.fc = torch.nn.Identity()
    feature_model.eval()
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    import cv2
    cap = cv2.VideoCapture(court_video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    features = []
    positions = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 15 == 0:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            img_t = transform(pil_img).unsqueeze(0)
            with torch.no_grad():
                feat = feature_model(img_t).squeeze().numpy()
            features.append(feat)
            positions.append(frame_idx)
        frame_idx += 1
    cap.release()
    
    features = np.array(features)
    positions = np.array(positions)
    
    # Define dummy segment boundaries (every 60 frames)
    shot_length = 60
    n_segs = int(np.ceil(frame_idx / shot_length))
    cps = []
    nfps = []
    for i in range(n_segs):
        start = i * shot_length
        end = min((i + 1) * shot_length - 1, frame_idx - 1)
        cps.append([start, end])
        nfps.append(end - start + 1)
    cps = np.array(cps)
    nfps = np.array(nfps)
    
    # Save into custom H5 file
    with h5py.File(h5_path, 'w') as h5:
        grp = h5.create_group('video_1')
        grp.create_dataset('features', data=features)
        grp.create_dataset('picks', data=positions)
        grp.create_dataset('n_frames', data=frame_idx)
        grp.create_dataset('change_points', data=cps)
        grp.create_dataset('n_frame_per_seg', data=nfps)
        
    print(f"Created court dataset H5 with {frame_idx} total frames and {features.shape[0]} feature vectors.")

    # 3. Quick training of sequence model (5 epochs)
    print("\nTraining LegalSum sequence model on court proceeding features...")
    model = DSN(in_dim=1024, hid_dim=256, num_layers=2, cell='lstm', num_heads=8, dropout=0.40)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()
    
    features_t = torch.from_numpy(features).unsqueeze(0).float() # [1, seq_len, 1024]
    
    # Simple training loop for policy gradient optimization
    for epoch in range(1, 6):
        optimizer.zero_grad()
        probs = model(features_t).squeeze() # [seq_len]
        
        # Policy gradient REINFORCE formulation
        dist = torch.distributions.Bernoulli(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        
        # Unsupervised reward: diversity + representativeness
        # Simplify reward computation for demonstration
        reward = float(actions.mean().item() * 0.5)
        loss = - (log_probs * reward).mean()
        loss.backward()
        optimizer.step()
        print(f"  Epoch {epoch}/5 - Policy Loss: {loss.item():.4f}")
        
    court_checkpoint = os.path.join(base_dir, 'demo/court_model_best.pth.tar')
    torch.save(model.state_dict(), court_checkpoint)
    print(f"Model trained and saved to: {court_checkpoint}")
    
    # 4. Generate Summarization
    court_summary = os.path.join(base_dir, 'demo/court_summary.mp4')
    court_manifest = os.path.join(base_dir, 'demo/court_audit_manifest.json')
    
    print("\nRunning multimodal narrative summarization...")
    run_legal_sum(court_video, court_summary, court_manifest, court_checkpoint, mode='narrative')
    
    print("==========================================================")

if __name__ == '__main__':
    main()
