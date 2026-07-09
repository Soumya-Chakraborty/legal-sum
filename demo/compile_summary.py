import os
import sys
import argparse
import json
import subprocess
import cv2
import numpy as np

# Add parent directory to path so we can import knapsack solver
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from knapsack import knapsack_dp

def load_cache(cache_path):
    with open(cache_path, 'r') as f:
        return json.load(f)

def compile_dynamic_summary(cache_path, video_path, output_video_path, target_duration_secs):
    print("==========================================================")
    print("       LEGALSUM: DYNAMIC LENGTH VIDEO COMPILER            ")
    print("==========================================================\n")
    
    # 1. Load pre-computed analysis cache
    cache = load_cache(cache_path)
    total_frames = cache["total_frames"]
    fps = cache["fps"]
    frame_scores = np.array(cache["frame_scores"])
    motion_scores = np.array(cache["motion_scores"])
    audio_loudness = np.array(cache["audio_loudness"])
    
    video_duration_secs = total_frames / fps
    print(f"Loaded court video metadata:")
    print(f"  Total Duration: {video_duration_secs:.2f}s ({total_frames} frames @ {fps} FPS)")
    print(f"  Target Duration: {target_duration_secs:.2f}s")
    
    if target_duration_secs >= video_duration_secs:
        print("Warning: Target duration exceeds video length. Outputting full video.")
        target_duration_secs = video_duration_secs
        
    # 2. Re-compute segment boundaries & anomalies
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
    
    # Fused anomalies
    multimodal_anomaly = 0.5 * motion_scores + 0.5 * audio_loudness
    
    # Segment-level anomalies for Action-Locking (Top 10%)
    seg_anomalies = []
    for i in range(n_segs):
        start, end = int(cps[i, 0]), int(cps[i, 1] + 1)
        seg_anomalies.append(float(multimodal_anomaly[start:end].mean()) if end > start else 0.0)
    seg_anomalies = np.array(seg_anomalies)
    
    anomaly_threshold = np.percentile(seg_anomalies, 90)
    locked_segs = []
    seg_scores = []
    
    legal_keywords = ["guilty", "confess", "murder", "verdict", "objection", "witness", "swear", "truth", "deny", "admit", "lie", "kill", "theft", "steal", "charge", "arrest", "conspiracy", "conspire", "court", "judge", "trial"]
    transcription_segments = cache.get("transcription_segments", [])
    
    for i in range(n_segs):
        start, end = int(cps[i, 0]), int(cps[i, 1] + 1)
        
        # Check semantic boost
        semantic_boost = 0.0
        for trans_seg in transcription_segments:
            overlap_start = max(start, trans_seg['start_frame'])
            overlap_end = min(end, trans_seg['end_frame'])
            if overlap_start < overlap_end:
                text_lower = trans_seg['text'].lower()
                if any(kw in text_lower for kw in legal_keywords):
                    semantic_boost = 0.35
                    break
        
        if seg_anomalies[i] > anomaly_threshold:
            locked_segs.append(i)
            seg_scores.append(1.0)
        else:
            base_score = float(frame_scores[start:end].mean())
            boosted_score = min(1.0, base_score + semantic_boost)
            if semantic_boost > 0:
                print(f"Applying semantic boost to segment {i} ({start}-{end}) due to legal keywords.")
            seg_scores.append(boosted_score)
            
    # 3. Solve Knapsack with target duration capacity
    target_frames_limit = int(target_duration_secs * fps)
    
    # Subtract locked segments from capacity if they are pre-selected
    locked_frames_sum = sum(nfps[locked_segs])
    
    print(f"Action-Lock pre-selected {len(locked_segs)} segments ({locked_frames_sum / fps:.2f}s).")
    
    # Filter out locked segments from knapsack solver to prevent duplicates
    dynamic_seg_scores = []
    dynamic_nfps = []
    dynamic_indices = []
    for i in range(n_segs):
        if i not in locked_segs:
            dynamic_seg_scores.append(seg_scores[i])
            dynamic_nfps.append(int(nfps[i]))
            dynamic_indices.append(i)
            
    # Calculate remaining capacity for dynamic selection
    remaining_capacity = int(max(0, target_frames_limit - locked_frames_sum))
    
    if remaining_capacity > 0 and len(dynamic_indices) > 0:
        picks_idx = knapsack_dp(dynamic_seg_scores, dynamic_nfps, len(dynamic_indices), remaining_capacity)
        picks = locked_segs + [dynamic_indices[p] for p in picks_idx]
    else:
        picks = locked_segs
        
    summary = np.zeros((total_frames,), dtype=np.int32)
    for seg_idx in range(n_segs):
        if seg_idx in picks:
            start, end = cps[seg_idx]
            summary[start:end+1] = 1
            
    # 4. Compile video slices using FFmpeg copy
    intervals = []
    if len(picks) > 0:
        picks.sort()
        start = cps[picks[0]][0]
        for i in range(1, len(picks)):
            if picks[i] != picks[i-1] + 1:
                intervals.append((start, cps[picks[i-1]][1]))
                start = cps[picks[i]][0]
        intervals.append((start, cps[picks[-1]][1]))

    print(f"\nSplicing {len(intervals)} video highlights via FFmpeg...")
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
            
    actual_duration_secs = float(np.sum(summary) / fps)
    print(f"\nSuccessfully generated dynamic summary: {output_video_path}")
    print(f"Actual Compiled Summary Duration: {actual_duration_secs:.2f}s ({np.sum(summary)} frames)")
    print("==========================================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LegalSum Dynamic Video Compiler")
    parser.add_argument('--cache', type=str, default='demo/court_analysis_cache.json', help='Path to precomputed features cache JSON')
    parser.add_argument('--input', type=str, default='demo/court_trial_naruto.webm', help='Path to master input video')
    parser.add_argument('--output', type=str, default='demo/court_summary_naruto.mp4', help='Path to output summary video')
    parser.add_argument('--duration', type=float, required=True, help='Target summary duration in seconds')
    
    args = parser.parse_args()
    compile_dynamic_summary(args.cache, args.input, args.output, args.duration)
