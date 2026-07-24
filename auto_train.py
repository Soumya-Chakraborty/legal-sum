#!/usr/bin/env python3
"""
auto_train.py — Hardware & Bottleneck-Aware Smart Training Launcher
===================================================================
Automatically inspects local system resources (CPU, GPU, VRAM, System RAM)
and dataset characteristics (sequence lengths T) to diagnose memory/compute
bottlenecks, then auto-tunes and executes main.py with parameters configured
for MAXIMUM F1 SCORE across any hardware environment.

Replaces train_cpu.py and train_gpu.py with a single intelligent launcher.

Usage:
    python3 auto_train.py -d datasets/eccv16_dataset_courtsum_google_pool5.h5 -s datasets/courtsum_splits.json -m tvsum
    python3 auto_train.py -d datasets/eccv16_dataset_summe_google_pool5.h5 -s datasets/splits/summe_splits.json -m summe
"""

import sys, os, os.path as osp, argparse, subprocess, h5py

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_system_specs():
    """Detect GPU, VRAM, System RAM, and CPU core count."""
    specs = {
        'cuda_available': False,
        'gpu_name': 'None',
        'vram_gb': 0.0,
        'cpu_cores': os.cpu_count() or 4,
        'total_ram_gb': 8.0,
        'avail_ram_gb': 6.0
    }

    # GPU / CUDA detection
    if HAS_TORCH and torch.cuda.is_available():
        specs['cuda_available'] = True
        specs['gpu_name'] = torch.cuda.get_device_name(0)
        specs['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    # System RAM detection (Linux /proc/meminfo)
    if osp.exists('/proc/meminfo'):
        try:
            with open('/proc/meminfo', 'r') as f:
                lines = f.readlines()
            mem_total = mem_avail = 0
            for line in lines:
                if line.startswith('MemTotal:'):
                    mem_total = int(line.split()[1]) / (1024 * 1024)
                elif line.startswith('MemAvailable:'):
                    mem_avail = int(line.split()[1]) / (1024 * 1024)
            specs['total_ram_gb'] = round(mem_total, 2)
            specs['avail_ram_gb'] = round(mem_avail, 2)
        except Exception:
            pass

    return specs

def analyze_dataset(dataset_path):
    """Analyze video sequence lengths (T) in the dataset."""
    info = {'total_videos': 0, 'max_T': 0, 'avg_T': 0, 'min_T': 0, 'is_long_sequence': False}
    if not osp.exists(dataset_path):
        return info

    try:
        with h5py.File(dataset_path, 'r') as h5:
            lengths = []
            for key in h5.keys():
                if 'features' in h5[key]:
                    lengths.append(h5[key]['features'].shape[0])
                elif 'n_frames' in h5[key]:
                    lengths.append(int(h5[key]['n_frames'][()]))
            if lengths:
                info['total_videos'] = len(lengths)
                info['max_T'] = max(lengths)
                info['min_T'] = min(lengths)
                info['avg_T'] = int(sum(lengths) / len(lengths))
                info['is_long_sequence'] = info['max_T'] > 3000
    except Exception as e:
        print(f"[WARN] Could not analyze dataset: {e}")

    return info

def main():
    parser = argparse.ArgumentParser("System & Bottleneck-Aware Smart Training Launcher (Max F1 Mode)")
    parser.add_argument('-d', '--dataset', type=str, required=True, help="Path to HDF5 dataset")
    parser.add_argument('-s', '--split', type=str, required=True, help="Path to splits JSON file")
    parser.add_argument('-m', '--metric', type=str, required=True, choices=['tvsum', 'summe'])
    parser.add_argument('--model-type', type=str, default='dual', choices=['original', 'enhanced', 'transformer', 'dual', 'legacy'],
                        help="Model architecture (default: dual for max multimodal F1)")
    parser.add_argument('--max-epoch', type=int, default=100, help="Total Phase-1 training epochs (default: 100)")
    parser.add_argument('--phase2-epochs', type=int, default=30, help="Phase-2 exploitation epochs (default: 30)")
    parser.add_argument('--pretrain-epochs', type=int, default=10, help="Phase-0 contrastive pretraining epochs (default: 10)")
    parser.add_argument('--save-dir', type=str, default='log/auto_max_f1')
    parser.add_argument('--max-seq-len', type=int, default=-1, help="Override max sequence length (-1 for auto)")
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--force-cpu', action='store_true', help="Force CPU training even if GPU is available")
    
    # Accept any extra args to forward to main.py
    parsed_args, extra_args = parser.parse_known_args()

    print("\n" + "═"*75)
    print("  LegalSum Hardware-Aware Auto-Launcher (MAX F1 OPTIMIZED)")
    print("═"*75)

    # 1. System Specs Diagnosis
    specs = get_system_specs()
    use_gpu = specs['cuda_available'] and not parsed_args.force_cpu

    print(f"\n[SYSTEM HARDWARE INSPECTION]")
    print(f"  • CPU Cores     : {specs['cpu_cores']} threads")
    print(f"  • System RAM    : {specs['total_ram_gb']} GB total ({specs['avail_ram_gb']} GB available)")
    print(f"  • GPU / CUDA    : {specs['gpu_name']} ({'AVAILABLE' if use_gpu else 'NOT AVAILABLE / DISABLED'})")
    if use_gpu:
        print(f"  • GPU VRAM      : {specs['vram_gb']:.2f} GB")

    # 2. Dataset Bottleneck Analysis
    ds_info = analyze_dataset(parsed_args.dataset)
    print(f"\n[DATASET BOTTLENECK ANALYSIS]")
    print(f"  • Dataset Path  : {parsed_args.dataset}")
    print(f"  • Total Videos  : {ds_info['total_videos']}")
    print(f"  • Sequence Lengths: T_min={ds_info['min_T']}, T_avg={ds_info['avg_T']}, T_max={ds_info['max_T']}")

    # 3. Bottleneck Diagnosis & Auto-Tuning
    print(f"\n[BOTTLENECK DIAGNOSIS & AUTO-TUNING FOR MAXIMUM F1]")
    auto_args = []

    # Sequence length ceiling tuning (prevents OOM explosion on multi-hour trials)
    if parsed_args.max_seq_len != -1:
        target_seq_len = parsed_args.max_seq_len
        print(f"  • Sequence Length : User override set to max_seq_len={target_seq_len}")
    elif ds_info['is_long_sequence']:
        if not use_gpu and specs['total_ram_gb'] < 16.0:
            target_seq_len = 2000
            print(f"  ⚠️ BOTTLENECK DETECTED: Long sequence (T={ds_info['max_T']}) on limited RAM ({specs['total_ram_gb']} GB).")
            print(f"    -> Auto-tuning --max-seq-len=2000 to prevent RAM OOM explosion while maintaining full temporal context.")
        elif use_gpu and specs['vram_gb'] < 10.0:
            target_seq_len = 2500
            print(f"  ⚠️ BOTTLENECK DETECTED: Long sequence (T={ds_info['max_T']}) on medium VRAM ({specs['vram_gb']:.1f} GB).")
            print(f"    -> Auto-tuning --max-seq-len=2500 for GPU memory safety.")
        else:
            target_seq_len = 0 # No limit on large systems
            print(f"  • Sequence Length : High memory available. Running 100% full sequence.")
    else:
        target_seq_len = 0 # Short datasets (SumMe/TVSum) remain untouched
        print(f"  • Sequence Length : Standard sequence (T_max={ds_info['max_T']}). Running 100% full sequence.")

    auto_args.extend(['--max-seq-len', str(target_seq_len)])

    # Model architecture & hyperparameter tuning for Maximum F1
    if not use_gpu and specs['total_ram_gb'] < 16.0:
        hidden_dim = 128
        num_heads = 4
        num_episode = 3
        ensemble_k = 5
        print(f"  • Hardware Profile: CPU Low-RAM Safe (hidden_dim=128, heads=4, episodes=3, K=5)")
    else:
        hidden_dim = 256
        num_heads = 8
        num_episode = 5
        ensemble_k = 10
        print(f"  • Hardware Profile: High Performance (hidden_dim=256, heads=8, episodes=5, K=10)")

    # Maximum F1 performance flags
    auto_args.extend(['--hidden-dim', str(hidden_dim)])
    auto_args.extend(['--num-heads', str(num_heads)])
    auto_args.extend(['--num-episode', str(num_episode)])
    auto_args.extend(['--ensemble-k', str(ensemble_k)])
    auto_args.extend(['--pretrain-epochs', str(parsed_args.pretrain_epochs)])
    auto_args.extend(['--ppo-clip', '0.2'])
    auto_args.extend(['--ppo-inner-steps', '4'])
    auto_args.extend(['--ot-weight', '0.10'])
    auto_args.extend(['--contrastive-weight', '0.05'])
    auto_args.extend(['--pr-f1-weight', '0.08'])
    auto_args.extend(['--recall-weight', '2.0'])
    auto_args.extend(['--tta-scales', '1.0,0.8,1.2'])

    # Hardware execution mode
    if use_gpu:
        auto_args.extend(['--gpu', parsed_args.gpu])
    else:
        auto_args.append('--use-cpu')

    # Construct final main.py command line
    cmd = [
        sys.executable, 'main.py',
        '-d', parsed_args.dataset,
        '-s', parsed_args.split,
        '-m', parsed_args.metric,
        '--model-type', parsed_args.model_type,
        '--max-epoch', str(parsed_args.max_epoch),
        '--phase2-epochs', str(parsed_args.phase2_epochs),
        '--save-dir', parsed_args.save_dir
    ] + auto_args + extra_args

    print("\n" + "═"*75)
    print("  Executing main.py with MAX F1 auto-tuned configuration:")
    print("  " + " ".join(cmd))
    print("═"*75 + "\n")

    # Execute main.py
    sys.exit(subprocess.call(cmd))

if __name__ == '__main__':
    main()
