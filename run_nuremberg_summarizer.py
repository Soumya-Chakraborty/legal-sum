import os
import sys

# Add parent directory to path so we can import from demo
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from demo.legal_sum import run_legal_sum

def main():
    video_path = 'demo/nuremberg_trial.mp4'
    output_video_path = 'demo/court_summary_nuremberg.mp4'
    manifest_path = 'demo/court_manifest_nuremberg.json'
    checkpoint_path = 'log/summe-counterfactual-optimized/model_best.pth.tar'

    run_legal_sum(
        video_path=video_path,
        output_video_path=output_video_path,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        mode='narrative',
        max_frames=None
    )

if __name__ == '__main__':
    main()
