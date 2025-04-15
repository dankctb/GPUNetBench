import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata 
from scipy.stats import norm
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# ——— Command‑line arguments —————————————————————————————————————
parser = argparse.ArgumentParser(
    description="Plot L2 slice bandwidth distribution from nvprof logs")
parser.add_argument(
    '--input-log', '-i',
    required=True,
    help="Path to the input log file (one bandwidth value per line, blank lines between slices)")
parser.add_argument(
    '--output', '-o',
    default='figure.png',
    help="Filename for the output plot image")
args = parser.parse_args()

# ——— Read the file and collect data ————————————————————————————————
z_dict = {}
with open(args.input_logs, 'r') as infile:
    slice_num = 0
    for line in infile:
        line = line.strip()
        if line == '':
            slice_num += 1
        else:
            bandwidth = float(line)
            z_dict.setdefault(slice_num, []).append(bandwidth)

# Flatten all values into one array
all_z = np.hstack(list(z_dict.values()))

# ——— Compute statistics ————————————————————————————————————————
median   = np.median(all_z)
avg      = np.mean(all_z)
std_dev  = np.std(all_z)

# ——— Automatic binning ————————————————————————————————————————
counts, bins = np.histogram(all_z, bins='auto')
counts = 100 * counts / counts.sum()  # convert to percentages

# ——— Plot setup ———————————————————————————————————————————————
fig, ax = plt.subplots(figsize=(4.5/2.54, 4/2.54))

# Draw the histogram
ax.bar(bins[:-1], counts, width=np.diff(bins),
       edgecolor='black', linewidth=1)

# Format y-axis as percentages
def to_percent(y, _):
    return f'{y:.0f}%'
ax.yaxis.set_major_formatter(FuncFormatter(to_percent))

ax.set_ylabel('Frequency')
ax.set_xlabel('Bandwidth (GB/s)')

# Automatically set x‑limits to data range
ax.set_xlim(all_z.min(), all_z.max())

# Add stats text
ax.text(
    0.02, 0.95,
    f'Average = {avg:.2f}\nStd Dev = {std_dev:.2f}',
    transform=ax.transAxes,
    verticalalignment='top'
)

# Save and show
plt.savefig(args.output, bbox_inches='tight', dpi=300)
plt.close(fig)
