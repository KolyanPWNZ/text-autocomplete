import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_training_history(history_df, save_path=None, figsize=(12, 5)):
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss
    ax = axes[0]
    ax.plot(history_df['epoch'], history_df['train_loss'], label='Train Loss', marker='o')
    if 'val_loss' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['val_loss'], label='Val Loss', marker='s')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Rouge
    ax = axes[1]
    if 'rouge1' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['rouge1'], label='ROUGE-1', marker='o')
    if 'rouge2' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['rouge2'], label='ROUGE-2', marker='s')
    if 'rougeL' in history_df.columns:
        ax.plot(history_df['epoch'], history_df['rougeL'], label='ROUGE-L', marker='^', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Rouge Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.5, history_df[['rouge1', 'rouge2', 'rougeL']].max().max() * 1.2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"График сохранён: {save_path}")
    
    plt.show()