import os
import sys
import time
import urllib.request
import cv2
import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as transforms
from PIL import Image

# Add parent directory to path so we can import DSN model and knapsack solver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models import DSN
from knapsack import knapsack_dp

def download_video(url, dest_path):
    print(f"Downloading sample video from: {url}")
    start = time.time()
    req = urllib.request.Request(
        url, 
        headers={'User-Agent': 'Mozilla/5.0'}
    )
    with urllib.request.urlopen(req) as response, open(dest_path, 'wb') as out_file:
        out_file.write(response.read())
    print(f"Download complete in {time.time() - start:.2f}s. Saved to {dest_path}")

def extract_googlenet_features(video_path, sample_rate=15):
    print("\nExtracting GoogLeNet features from video...")
    # Load GoogLeNet with pre-trained weights
    # We use weights=GoogLeNet_Weights.DEFAULT or pretrained=True depending on torchvision version
    try:
        weights = tv_models.GoogLeNet_Weights.DEFAULT
        model = tv_models.googlenet(weights=weights)
    except AttributeError:
        model = tv_models.googlenet(pretrained=True)
    
    # We need the output of the pool5 layer (1024-dim representation)
    # Remove the classification head by converting the fc layer to identity
    model.fc = torch.nn.Identity()
    model.eval()
    
    # Image transformations matching GoogLeNet inputs
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video stats: {total_frames} frames, {fps:.2f} FPS, {width}x{height}")
    
    features = []
    positions = []
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Sample frames at the specified sample_rate (typically pick 1 frame every 15 frames)
        if frame_idx % sample_rate == 0:
            # Convert CV2 BGR to PIL RGB Image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img)
            
            # Apply preprocessing
            img_t = transform(pil_img).unsqueeze(0)
            
            # Extract feature vector
            with torch.no_grad():
                feat = model(img_t).squeeze().numpy()
            features.append(feat)
            positions.append(frame_idx)
            
        frame_idx += 1
        
    cap.release()
    features = np.array(features)
    positions = np.array(positions)
    print(f"Extracted {features.shape[0]} feature vectors of dimension {features.shape[1]}")
    return features, positions, total_frames, fps, width, height

def generate_custom_summary(video_path, output_summary_path, checkpoint_path):
    # 1. Extract GoogLeNet pooling features
    features, positions, total_frames, fps, width, height = extract_googlenet_features(video_path)
    
    # 2. Load our trained SOTA optimized model
    print("\nLoading SCSL-SGC video summarization model...")
    model = DSN(in_dim=1024, hid_dim=256, num_layers=2, cell='lstm', num_heads=8, dropout=0.40)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    # 3. Predict frame importance scores
    features_t = torch.from_numpy(features).unsqueeze(0).float()
    print("Running sequence model inference...")
    with torch.no_grad():
        probs = model(features_t).squeeze().numpy()
    
    # 4. Generate pseudo-shots (since we don't have human change points)
    # Define a shot as 60 consecutive frames
    shot_length = 60
    n_segs = int(np.ceil(total_frames / shot_length))
    cps = []
    nfps = []
    for i in range(n_segs):
        start = i * shot_length
        end = min((i + 1) * shot_length - 1, total_frames - 1)
        cps.append([start, end])
        nfps.append(end - start + 1)
    cps = np.array(cps)
    nfps = np.array(nfps)
    
    # Map predictions back to original frames
    frame_scores = np.zeros((total_frames,), dtype=np.float32)
    # Ensure boundary conditions include the very end of the video
    padded_positions = np.concatenate([positions, [total_frames]])
    for i in range(len(padded_positions) - 1):
        pos_left, pos_right = padded_positions[i], padded_positions[i + 1]
        if i == len(probs):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = probs[i]
            
    # Calculate segment average scores
    seg_scores = []
    for i in range(n_segs):
        start, end = int(cps[i, 0]), int(cps[i, 1] + 1)
        seg_scores.append(float(frame_scores[start:end].mean()))
        
    # Solve 0/1 Knapsack to select shots under 15% length limit
    limit = int(total_frames * 0.15)
    picks = knapsack_dp(seg_scores, nfps.tolist(), n_segs, limit)
    
    # Construct binary frame-level selection array
    summary = np.zeros((total_frames,), dtype=np.float32)
    for seg_idx in range(n_segs):
        if seg_idx in picks:
            start, end = cps[seg_idx]
            summary[start:end+1] = 1
            
    # 5. Compile summary video using OpenCV
    print(f"\nCompiling summary video: keeping {int(summary.sum())} frames out of {total_frames}...")
    cap = cv2.VideoCapture(video_path)
    
    # Configure output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(output_summary_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    written_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(summary) and summary[frame_idx] == 1:
            out_writer.write(frame)
            written_count += 1
        frame_idx += 1
        
    cap.release()
    out_writer.release()
    print(f"Summary video successfully written to: {output_summary_path}")
    print(f"Final summary duration: {written_count/fps:.2f} seconds.")

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    video_url = "https://upload.wikimedia.org/wikipedia/commons/f/fb/2022_FIFA_World_Cup%27s_first_goal_by_Enner_Valencia_of_Ecuador_against_Qatar.webm"
    input_video = os.path.join(base_dir, 'demo/fifa_match.webm')
    output_summary = os.path.join(base_dir, 'demo/fifa_summary.mp4')
    checkpoint = os.path.join(base_dir, 'log/summe-counterfactual-optimized/model_best.pth.tar')
    
    # Download sample video
    download_video(video_url, input_video)
    
    # Run summarization
    generate_custom_summary(input_video, output_summary, checkpoint)
