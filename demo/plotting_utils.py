import os
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt

def plot_training_curves(history, save_dir):
    """
    Plots and saves separate training performance curves.
    
    Args:
        history (dict): Dictionary containing lists of metrics:
            - 'epoch': list of epochs
            - 'f_score': list of F-scores
            - 'spearman': list of Spearman correlations
            - 'kendall': list of Kendall correlations
            - 'event_coverage': list of Event Coverages
            - 'speaker_consistency': list of Speaker Turn Consistencies
            - 'reward': list of training rewards
            - 'entropy': list of policy entropies
        save_dir (str): Directory where plots will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = history.get('epoch', [])
    if not epochs:
        return
        
    # Style configuration
    plt.style.use('seaborn-v0_8-paper' if 'seaborn-v0_8-paper' in plt.style.available else 'default')
    
    # 1. F-Score Curve
    if 'f_score' in history and history['f_score']:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, [x * 100 for x in history['f_score']], 'b-o', label='F-Score (%)')
        plt.xlabel('Epoch')
        plt.ylabel('F-Score (%)')
        plt.title('F-Score progress')
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'f_score_curve.png'), dpi=300)
        plt.close()
        
    # 2. Correlation Curves
    if 'spearman' in history and 'kendall' in history:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, history['spearman'], 'r-s', label='Spearman Corr')
        plt.plot(epochs, history['kendall'], 'g-^', label='Kendall Corr')
        plt.xlabel('Epoch')
        plt.ylabel('Correlation')
        plt.title('Rank Correlation against human annotations')
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'correlation_curve.png'), dpi=300)
        plt.close()
        
    # 3. Courtroom Specific Coverage Curves
    if 'event_coverage' in history and 'speaker_consistency' in history:
        plt.figure(figsize=(6, 4))
        plt.plot(epochs, [x * 100 for x in history['event_coverage']], 'm-d', label='Event Coverage (%)')
        plt.plot(epochs, [x * 100 for x in history['speaker_consistency']], 'c-x', label='Speaker Consistency (%)')
        plt.xlabel('Epoch')
        plt.ylabel('Coverage / Consistency (%)')
        plt.title('Courtroom specific objectives')
        plt.grid(True, linestyle='--')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'courtroom_coverage_curve.png'), dpi=300)
        plt.close()
        
    # 4. Training Rewards & Policy Entropy Curves
    if 'reward' in history and 'entropy' in history:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        
        color = 'tab:blue'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Mean Reward', color=color)
        ax1.plot(epochs, history['reward'], color=color, marker='o', label='Reward')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.grid(True, linestyle='--')
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Policy Entropy', color=color)
        ax2.plot(epochs, history['entropy'], color=color, marker='s', label='Entropy')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Training Reward and Policy Entropy')
        fig.tight_layout()
        plt.savefig(os.path.join(save_dir, 'reward_entropy_curve.png'), dpi=300)
        plt.close()
        
    print(f"Separate computation plots generated successfully in: {save_dir}")
