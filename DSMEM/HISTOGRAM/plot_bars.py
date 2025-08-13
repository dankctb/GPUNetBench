from pathlib import Path
import pandas as pd, matplotlib.pyplot as plt

p = Path(__file__).resolve().parent
df = pd.read_csv(p / 'output.csv')
ax = df.pivot(index='bin_size', columns='cluster_size', values='kernel_execution_time(ms)').plot(kind='bar', figsize=(8, 4))
ax.set_xlabel('bins size (#elements)'); ax.set_ylabel('Kernel execution time (ms)'); ax.set_title('Histogram')
ax.legend(title='cluster size', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.xticks(rotation=0); plt.tight_layout(); plt.savefig(p / 'output_bars.png', dpi=150, bbox_inches='tight') 