#!/usr/bin/env python3
"""
run_all_experiments.py
Full 5-fold cross-validation on SumMe and TVSum.
Prints per-split and mean F-scores to stdout.
"""
import os, sys, subprocess, json, time, datetime

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

def run_split(dataset_name, split_id, cfg):
    tag = f'{dataset_name}-split{split_id}'
    save_dir = f'{BASE}/log/exp_cv/{tag}'
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        PYTHON, f'{BASE}/main.py',
        '-d', cfg['h5'],
        '-s', cfg['split'],
        '--split-id', str(split_id),
        '-m', cfg['metric'],
        '--model-type', 'enhanced',
        '--hidden-dim', '256',
        '--num-layers', '2',
        '--num-heads', '8',
        '--dropout', '0.25',
        '--lr', '1e-4',
        '--weight-decay', '1e-5',
        '--max-epoch', '100',
        '--phase2-epochs', '30',
        '--num-episode', '5',
        '--entropy-start', '0.10',
        '--entropy-end', '0.001',
        '--use-counterfactual',
        '--ensemble-k', '10',
        '--seed', str(42 + split_id),
        '--use-cpu',          # CPU-only (no GPU in env)
        '--save-dir', save_dir,
        '--save-results',
        '--verbose',
    ]

    print(f'\n[{tag}] Starting...', flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
    elapsed = str(datetime.timedelta(seconds=round(time.time() - t0)))

    # Parse final F-score from stdout
    fm = None
    for line in proc.stdout.splitlines():
        if 'OVERALL BEST F-score' in line:
            try:
                fm = float(line.split(':')[-1].strip().rstrip('%')) / 100.0
            except:
                pass
        if 'Average F-score' in line and 'epoch' not in line.lower():
            try:
                fm_candidate = float(line.split('Average F-score')[1].strip().split('%')[0]) / 100.0
                if fm is None:
                    fm = fm_candidate
            except:
                pass

    # Save full logs
    with open(f'{save_dir}/stdout.txt', 'w') as f:
        f.write(proc.stdout)
    with open(f'{save_dir}/stderr.txt', 'w') as f:
        f.write(proc.stderr)

    print(f'[{tag}] Done in {elapsed}. F-score = {fm:.1%}' if fm else f'[{tag}] Done in {elapsed}. F-score parse failed', flush=True)
    return fm


def main():
    results = {}
    for ds_name, cfg in DATASETS.items():
        print(f'\n{"="*60}')
        print(f'Dataset: {ds_name.upper()}  ({cfg["splits"]}-fold CV)')
        print(f'{"="*60}')
        fms = []
        for sid in range(cfg['splits']):
            fm = run_split(ds_name, sid, cfg)
            fms.append(fm)
            print(f'  Split {sid}: {fm:.1%}' if fm is not None else f'  Split {sid}: ERROR')

        valid = [f for f in fms if f is not None]
        mean_fm = sum(valid) / len(valid) if valid else 0.0
        results[ds_name] = {'per_split': fms, 'mean': mean_fm}
        print(f'\n{ds_name.upper()} Mean F-score: {mean_fm:.1%}')

    print('\n' + '='*60)
    print('FINAL RESULTS (5-fold cross-validation)')
    print('='*60)
    for ds_name, r in results.items():
        splits_str = ', '.join(f'{f:.1%}' if f else 'ERR' for f in r['per_split'])
        print(f'{ds_name.upper():8s}: [{splits_str}]  mean = {r["mean"]:.1%}')

    # Save results JSON
    out = f'{BASE}/log/exp_cv/cv_results.json'
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to: {out}')


if __name__ == '__main__':
    main()
