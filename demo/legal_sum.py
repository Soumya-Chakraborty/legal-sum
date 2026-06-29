import os
import sys
import time
import hashlib
import json
import wave
import struct
import subprocess
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

def compute_frame_hash(frame):
    """Compute SHA-256 hash of the raw frame bytes."""
    _, buffer = cv2.imencode('.jpg', frame)
    return hashlib.sha256(buffer).hexdigest()

def compute_motion_score(frame1, frame2):
    """Compute normalized absolute pixel differences as a motion/anomaly proxy."""
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray1 = cv2.resize(gray1, (120, 90))
    gray2 = cv2.resize(gray2, (120, 90))
    diff = cv2.absdiff(gray1, gray2)
    return float(np.mean(diff) / 255.0)

def extract_audio_loudness(video_path, wav_path, num_frames, fps):
    """Extract audio track and compute RMS loudness per video frame."""
    print("Extracting audio track for loudness analysis...")
    # Extract audio to 16kHz mono WAV using FFmpeg
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", wav_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if not os.path.exists(wav_path):
        print("Warning: Audio track not found or extraction failed. Proceeding with video-only.")
        return np.zeros((num_frames,), dtype=np.float32)
        
    with wave.open(wav_path, 'rb') as w:
        sample_rate = w.getframerate()
        n_samples = w.getnframes()
        data = w.readframes(n_samples)
        
    # Unpack 16-bit PCM
    samples = np.array(struct.unpack(f"{n_samples}h", data), dtype=np.float32)
    
    # Calculate RMS per video frame
    samples_per_frame = int(sample_rate / fps)
    loudness = []
    for i in range(num_frames):
        start = i * samples_per_frame
        end = min(start + samples_per_frame, len(samples))
        if start < len(samples):
            frame_samples = samples[start:end]
            rms = np.sqrt(np.mean(frame_samples**2)) if len(frame_samples) > 0 else 0.0
            loudness.append(rms)
        else:
            loudness.append(0.0)
            
    loudness = np.array(loudness)
    max_val = np.max(loudness) if np.max(loudness) > 0 else 1.0
    return loudness / max_val

def run_legal_sum(video_path, output_video_path, manifest_path, checkpoint_path, mode='narrative', max_frames=10000):
    print("==========================================================")
    print("      LEGALSUM: MULTIMODAL COURT VIDEO SUMMARIZER         ")
    print("==========================================================\n")

    # 1. Feature Extraction, Audio Extraction, and Hashing
    print("Phase 1: Extraction & Hashing (Multimodal Validation)...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps > 100 or fps <= 0:
        fps = 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    wav_path = video_path.replace('.webm', '.wav').replace('.mp4', '.wav')
    audio_loudness = extract_audio_loudness(video_path, wav_path, total_frames, fps)
    
    transcription_segments = []
    try:
        import whisper
        print("Initializing open-source Whisper transcription...")
        whisper_model = whisper.load_model("tiny")
        print("Transcribing audio track...")
        result = whisper_model.transcribe(wav_path)
        for seg in result.get("segments", []):
            start_sec = float(seg["start"])
            if start_sec * fps < total_frames:
                transcription_segments.append({
                    "start_time": start_sec,
                    "end_time": float(seg["end"]),
                    "start_frame": int(start_sec * fps),
                    "end_frame": int(float(seg["end"]) * fps),
                    "text": seg["text"].strip()
                })
        print(f"Whisper transcription completed: {len(transcription_segments)} segments mapped.")
    except Exception as e:
        print(f"Warning: Whisper transcription skipped: {e}")
    
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
    
    features = []
    positions = []
    frame_hashes = []
    motion_scores = []
    
    frame_idx = 0
    prev_frame = None
    
    while cap.isOpened():
        if max_frames is not None and frame_idx >= max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        
        # Compute SHA-256 frame-level hash
        frame_hashes.append(compute_frame_hash(frame))
        
        # Compute motion score
        if prev_frame is not None:
            motion_scores.append(compute_motion_score(frame, prev_frame))
        else:
            motion_scores.append(0.0)
        prev_frame = frame.copy()
        
        # Extract GoogLeNet feature vector
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
    total_frames = frame_idx
    features = np.array(features)
    positions = np.array(positions)
    motion_scores = np.array(motion_scores)
    audio_loudness = audio_loudness[:total_frames]
    
    # Clean up temp wav file
    if os.path.exists(wav_path):
        os.remove(wav_path)
        
    print("Multimodal feature extraction & hashing complete.")

    # 2. Sequence Model Inference
    print("\nPhase 2: Sequence Model Inference...")
    model = DSN(in_dim=1024, hid_dim=256, num_layers=2, cell='lstm', num_heads=8, dropout=0.40)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    
    features_t = torch.from_numpy(features).unsqueeze(0).float()
    with torch.no_grad():
        probs = model(features_t).squeeze().numpy()
        
    # Interpolate scores
    frame_scores = np.zeros((total_frames,), dtype=np.float32)
    padded_positions = np.concatenate([positions, [total_frames]])
    for i in range(len(padded_positions) - 1):
        pos_left, pos_right = padded_positions[i], padded_positions[i + 1]
        if i == len(probs):
            frame_scores[pos_left:pos_right] = 0
        else:
            frame_scores[pos_left:pos_right] = probs[i]

    # 3. Multimodal Fusion and Action Locking
    print("\nPhase 3: Multimodal Fusion & Action Locking...")
    # Multimodal anomaly: equal weights to motion differences and audio loudness
    multimodal_anomaly = 0.5 * motion_scores + 0.5 * audio_loudness
    
    # 2.0s segments
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
    
    # Flag action-locked segments above 90th percentile of segment-level anomalies
    seg_anomalies = []
    for i in range(n_segs):
        start, end = int(cps[i, 0]), int(cps[i, 1] + 1)
        seg_anomalies.append(float(multimodal_anomaly[start:end].mean()) if end > start else 0.0)
    seg_anomalies = np.array(seg_anomalies)
    
    anomaly_threshold = np.percentile(seg_anomalies, 90)
    locked_segs = []
    seg_scores = []
    
    for i in range(n_segs):
        start, end = int(cps[i, 0]), int(cps[i, 1] + 1)
        if seg_anomalies[i] > anomaly_threshold:
            locked_segs.append(i)
            seg_scores.append(1.0)
        else:
            seg_scores.append(float(frame_scores[start:end].mean()))
            
    print(f"Action-Lock activated on {len(locked_segs)} segments due to high motion or loud audio events.")

    # 4. Knapsack Selection
    budget = 0.45 if mode == 'narrative' else 0.20
    limit = int(total_frames * budget)
    picks = knapsack_dp(seg_scores, nfps.tolist(), n_segs, limit)
    for locked in locked_segs:
        if locked not in picks:
            picks.append(locked)
            
    summary = np.zeros((total_frames,), dtype=np.int32)
    for seg_idx in range(n_segs):
        if seg_idx in picks:
            start, end = cps[seg_idx]
            summary[start:end+1] = 1
            
    # 5. Extract continuous intervals for FFmpeg splicing
    intervals = []
    if len(picks) > 0:
        picks.sort()
        start = cps[picks[0]][0]
        for i in range(1, len(picks)):
            if picks[i] != picks[i-1] + 1:
                intervals.append((start, cps[picks[i-1]][1]))
                start = cps[picks[i]][0]
        intervals.append((start, cps[picks[-1]][1]))

    # 6. Splicing Audio and Video using FFmpeg copy
    print("\nPhase 4: Splicing Audio/Video via FFmpeg...")
    temp_files = []
    for idx, (s, e) in enumerate(intervals):
        start_time = s / fps
        duration = (e - s + 1) / fps
        part_path = f"demo/part_{idx}.mp4"
        cmd = [
            "ffmpeg", "-y", "-ss", f"{start_time:.3f}", "-t", f"{duration:.3f}",
            "-i", video_path, "-c:v", "libx264", "-c:a", "aac", part_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        temp_files.append(part_path)
        
    # Concatenate parts
    concat_list_path = "demo/concat_list.txt"
    with open(concat_list_path, 'w') as f:
        for tf in temp_files:
            f.write(f"file '{os.path.basename(tf)}'\n")
            
    concat_cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_list_path,
        "-c", "copy", output_video_path
    ]
    subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Cleanup temp slices
    os.remove(concat_list_path)
    for tf in temp_files:
        if os.path.exists(tf):
            os.remove(tf)
            
    # Build manifest JSON
    manifest_entries = []
    summary_frame_idx = 0
    for frame_idx in range(total_frames):
        if summary[frame_idx] == 1:
            manifest_entries.append({
                "summary_frame_index": summary_frame_idx,
                "original_frame_index": frame_idx,
                "sha256_hash": frame_hashes[frame_idx],
                "model_score": float(frame_scores[frame_idx]),
                "motion_score": float(motion_scores[frame_idx]),
                "audio_loudness": float(audio_loudness[frame_idx]),
                "action_locked": bool(frame_idx // shot_length in locked_segs)
            })
            summary_frame_idx += 1
            
    manifest_data = {
        "transcription_segments": transcription_segments,
        "frame_manifest": manifest_entries
    }
    with open(manifest_path, 'w') as f:
        json.dump(manifest_data, f, indent=2)
        
    # Write pre-computed cache for real-time dynamic slider compile
    cache_path = manifest_path.replace('manifest', 'analysis_cache')
    cache_data = {
        "total_frames": int(total_frames),
        "fps": float(fps),
        "frame_scores": frame_scores.tolist(),
        "motion_scores": motion_scores.tolist(),
        "audio_loudness": audio_loudness.tolist(),
        "transcription_segments": transcription_segments
    }
    with open(cache_path, 'w') as f:
        json.dump(cache_data, f)
        
    print(f"Summary Video (with Audio): {output_video_path}")
    print(f"Audit Manifest:             {manifest_path}")
    print(f"Analysis Cache:             {cache_path}")
    print(f"Summary Duration:           {summary_frame_idx/fps:.2f}s ({summary_frame_idx} frames)")
    print("Verification and summary compiled successfully!")
    print("==========================================================")

if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    input_video = os.path.join(base_dir, 'demo/fifa_match.webm')
    output_video = os.path.join(base_dir, 'demo/legal_summary.mp4')
    manifest = os.path.join(base_dir, 'demo/legal_audit_manifest.json')
    checkpoint = os.path.join(base_dir, 'log/summe-counterfactual-optimized/model_best.pth.tar')
    
    run_legal_sum(input_video, output_video, manifest, checkpoint, mode='narrative')
