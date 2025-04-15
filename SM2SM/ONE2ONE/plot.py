#!/usr/bin/env python3
import argparse
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

# Set the global font to DejaVu Sans, size 14
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ——— Command‑line arguments —————————————————————————————————————
parser = argparse.ArgumentParser(
    description="Plot bandwidth distribution from a hist file (bw mode only)")
parser.add_argument(
    '--input-log', '-i',
    required=True,
    help="Path to the input histogram file (one float per line, output of the bw hist script)")
parser.add_argument(
    '--output', '-o',
    default='figure.png',
    help="Filename for the output plot image")
args = parser.parse_args()

# ——— Read the file and collect data ————————————————————————————————
with open(args.input_log, 'r') as infile:
    # Read all non-empty lines as floats
    all_z = np.array([float(line.strip()) for line in infile if line.strip()])

# ——— Compute statistics ————————————————————————————————————————
avg     = np.mean(all_z)
std_dev = np.std(all_z)

# ——— Automatic binning over exact data range —————————————————————————
counts, bins = np.histogram(
    all_z,
    bins='auto',
    range=(all_z.min(), all_z.max())   # force bins to start/end at data min/max
)
counts = 100 * counts / counts.sum()   # convert to percentages

# ——— Plot setup —————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))

# Draw the histogram
ax.bar(bins[:-1], counts, width=np.diff(bins), edgecolor='black', linewidth=1)

# Format y-axis as percentages
def to_percent(y, _):
    return f'{y:.0f}%'
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

ax.set_ylabel('Frequency')
ax.set_xlabel('Bandwidth (GB/s)')

# Align x‑axis limits with the first and last bin edges
ax.set_xlim(bins[0]-0.1, bins[-1]+0.1)

# Add stats text
ax.text(
    0.02, 0.95,
    f'Average = {avg:.2f}\nStd Dev = {std_dev:.2f}',
    transform=ax.transAxes,
    verticalalignment='top'
)

plt.savefig(args.output, bbox_inches='tight', dpi=300)
plt.close(fig)
