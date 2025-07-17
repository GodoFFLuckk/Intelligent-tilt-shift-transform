import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('train_log.txt')
df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
for col in ['train_loss', 'val_loss', 'train_l1', 'val_l1', 'train_ssim', 'val_ssim', 'train_lpips', 'val_lpips']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

metrics = [
    ('train_loss', 'val_loss'),
    ('train_l1', 'val_l1'),
    ('train_ssim', 'val_ssim'),
    ('train_lpips', 'val_lpips'),
]
titles = [
    'Total Loss',
    'L1 Loss',
    'SSIM Loss',
    'LPIPS Loss'
]

step = 5
df_plot = df[df['epoch'] % step == 0].copy()

epochs = df_plot['epoch']

plt.figure(figsize=(14, 10))
for i, (train_col, val_col) in enumerate(metrics, 1):
    plt.subplot(2, 2, i)
    plt.plot(epochs, df_plot[train_col], label='Train', marker='o', linewidth=2)
    plt.plot(epochs, df_plot[val_col], label='Val', marker='s', linewidth=2, alpha=0.8)
    plt.title(titles[i-1])
    plt.xlabel('Epoch')
    plt.ylabel(titles[i-1])
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.xticks(np.arange(step, int(epochs.max())+1, step))
    plt.tight_layout()

plt.savefig('metrics.jpg', dpi=200)
plt.show()
