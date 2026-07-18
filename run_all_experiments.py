#!/usr/bin/env python3
"""
run_all_experiments.py
Full 5-fold cross-validation on SumMe and TVSum with multiple seeds and ablation tables.
"""
import os
import sys
import subprocess
import json
import time
import datetime
import argparse
import numpy as np
import re

BASE = '/home/developer/Personal/Unsupervised-video-summarization'
PYTHON = sys.executable

DATASETS = {
    'summe': {
        'h5':    f'{BASE}/datasets/eccv16_dataset_summe_google_pool5.h5',
        'split': f'{BASE}/datasets/summe_splits.json',
        'metric': 'summe',
        'splits': 5,
    },
    'tvsum': {
        'h5':    f'{BASE}/datasets/eccv16_dataset_tvsum_google_pool5.h5',
        'split': f'{BASE}/datasets/tvsum_splits.json',
        'metric': 'tvsum',
        'splits': 5,
    },
}

CONFIGS = {
    'baseline': [
        '--model-type', 'original',
        '--no-counterfactual',
        '--contrastive-weight', '0.0',
    ],
    'ours': [
        '--model-type', 'enhanced',
        '--use-counterfactual',
        '--contrastive-weight', '0.05',
    ]
}

def run_experiment(dataset_name, split_id, seed, config_name, config_flags, args):
    tag = f'{dataset_name}-{config_name}-split{split_id}-seed{seed}'
    save_dir = f'{BASE}/log/exp_cv/{tag}'
    os.makedirs(save_dir, exist_ok=True)

    cfg = DATASETS[dataset_name]

    cmd = [
        PYTHON, f'{BASE}/main.py',
        '-d', cfg['h5'],
        '-s', cfg['split'],
        '--split-id', str(split_id),
        '-m', cfg['metric'],
        '--hidden-dim', '256',
        '--num-layers', '2',
        '--num-heads', '8',
        '--dropout', '0.25',
        '--lr', '1e-4',
        '--weight-decay', '1e-5',
        '--max-epoch', str(args.max_epoch),
        '--phase2-epochs', str(args.phase2_epochs),
        '--patience', str(args.patience),
        '--num-episode', '5',
        '--entropy-start', '0.10',
        '--entropy-end', '0.001',
        '--ensemble-k', '10',
        '--seed', str(seed),
        '--use-cpu',
        '--save-dir', save_dir,
    ] + config_flags

    if args.verbose:
        cmd.append('--verbose')

    print(f'\n[{tag}] Starting...', flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    elapsed = str(datetime.timedelta(seconds=round(time.time() - t0)))

    # Parse final F-scores from stdout
    f1_mean_best = None
    f1_max_best = None
    
    f1_mean_pattern = re.compile(r"Average F1\s*\(mean:\s*([\d.]+)%,\s*max:\s*([\d.]+)%\)")
    
    for line in proc.stdout.splitlines():
        match = f1_mean_pattern.search(line)
        if match:
            f1_mean = float(match.group(1)) / 100.0
            f1_max = float(match.group(2)) / 100.0
            if f1_mean_best is None or f1_mean > f1_mean_best:
                f1_mean_best = f1_mean
            if f1_max_best is None or f1_max > f1_max_best:
                f1_max_best = f1_max

    # Also check fallback "OVERALL BEST F-score" or "Average F-score"
    for line in proc.stdout.splitlines():
        if 'OVERALL BEST F-score' in line:
            try:
                fm = float(line.split(':')[-1].strip().rstrip('%')) / 100.0
                if f1_mean_best is None:
                    f1_mean_best = fm
                if f1_max_best is None:
                    f1_max_best = fm
            except:
                pass

    # Save full logs
    with open(f'{save_dir}/stdout.txt', 'w') as f:
        f.write(proc.stdout)
    with open(f'{save_dir}/stderr.txt', 'w') as f:
        f.write(proc.stderr)

    f1_mean_str = f'{f1_mean_best:.1%}' if f1_mean_best is not None else 'N/A'
    f1_max_str = f'{f1_max_best:.1%}' if f1_max_best is not None else 'N/A'
    print(f'[{tag}] Done in {elapsed}. F1-mean = {f1_mean_str}, F1-max = {f1_max_str}', flush=True)
    return f1_mean_best, f1_max_best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-epoch', type=int, default=100)
    parser.add_argument('--phase2-epochs', type=int, default=30)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seeds', type=str, default='42,43,44')
    parser.add_argument('--configs', type=str, default='baseline,ours')
    parser.add_argument('--splits', type=int, default=5)
    parser.add_argument('--datasets', type=str, default='summe,tvsum')
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(',') if s.strip()]
    configs_to_run = [c.strip() for c in args.configs.split(',') if c.strip()]
    datasets_to_run = [d.strip() for d in args.datasets.split(',') if d.strip()]
    num_splits = args.splits

    results = {}

    for ds_name in datasets_to_run:
        if ds_name not in DATASETS:
            continue
        results[ds_name] = {}
        for cfg_name in configs_to_run:
            if cfg_name not in CONFIGS:
                continue
            
            print(f'\n{"="*80}')
            print(f'Running: Dataset={ds_name.upper()}, Config={cfg_name.upper()}, Splits={num_splits}, Seeds={seeds}')
            print(f'{"="*80}')
            
            run_runs = []
            for sid in range(num_splits):
                split_runs = []
                for seed in seeds:
                    f_mean, f_max = run_experiment(ds_name, sid, seed, cfg_name, CONFIGS[cfg_name], args)
                    split_runs.append((f_mean, f_max))
                run_runs.append(split_runs)
            
            results[ds_name][cfg_name] = run_runs

    # Compile and Print Ablation / Comparison Table
    print('\n' + '='*80)
    print('ABLATION AND EXPERIMENT RESULTS COMPARISON')
    print('='*80)
    
    for ds_name, ds_res in results.items():
        print(f'\nDataset: {ds_name.upper()}')
        print('-'*80)
        print(f'{"Configuration":<20} | {"Mean F1 (avg)":<18} | {"Mean F1 (max)":<18}')
        print('-'*80)
        
        for cfg_name, runs in ds_res.items():
            all_means = []
            all_maxs = []
            for split_idx, split_runs in enumerate(runs):
                for seed_idx, (f_mean, f_max) in enumerate(split_runs):
                    if f_mean is not None:
                        all_means.append(f_mean)
                    if f_max is not None:
                        all_maxs.append(f_max)
            
            mean_avg = np.mean(all_means) if all_means else 0.0
            std_avg = np.std(all_means) if all_means else 0.0
            
            mean_max = np.mean(all_maxs) if all_maxs else 0.0
            std_max = np.std(all_maxs) if all_maxs else 0.0
            
            print(f'{cfg_name:<20} | {mean_avg:.1%} ± {std_avg:.1%}          | {mean_max:.1%} ± {std_max:.1%}')
        print('-'*80)

    # Save results JSON
    out_dir = f'{BASE}/log/exp_cv'
    os.makedirs(out_dir, exist_ok=True)
    out_path = f'{out_dir}/cv_results.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nAggregated results saved to: {out_path}')


if __name__ == '__main__':
    main()
