import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import argparse

# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

def main():
    parser = argparse.ArgumentParser(
        description="Generate combined histograms from three latency data files."
    )
    parser.add_argument("--files", nargs=3, default=["1_all.log", "32_all.log", "64_all.log"],
                        help="Three input files to load (default: 1_all.log 32_all.log 64_all.log)")
    parser.add_argument("--legends", nargs=3, default=["1 Threads", "32 Threads", "64 Threads"],
                        help="Three legend labels (default: '1 Threads', '32 Threads', '64 Threads')")
    parser.add_argument("--binstep", type=float, default=10,
                        help="Bin step size (default: 10)")
    parser.add_argument("--xrange", nargs=2, type=float, default=[100, 500],
                        help="X-axis range as two numbers (default: 100 500)")
    parser.add_argument("--ylim", type=float, default=30,
                        help="Y-axis upper limit in percent (default: 30)")
    parser.add_argument("--plot_filename", type=str, default="histogram.png",
                        help="Output file for the histogram (default: histogram.png)")

    args = parser.parse_args()
    
    # Load the three datasets.
    try:
        data1 = np.loadtxt(args.files[0])
        data2 = np.loadtxt(args.files[1])
        data3 = np.loadtxt(args.files[2])
    except Exception as e:
        sys.exit(f"Error loading one of the files: {e}")
    
    # Compute statistics.
    def compute_stats(data):
        return {
            'Mean': np.mean(data),
            'Std': np.std(data),
            'Max': np.max(data),
            'Min': np.min(data)
        }
    
    stats1 = compute_stats(data1)
    stats2 = compute_stats(data2)
    stats3 = compute_stats(data3)
    
    # Create figure.
    plt.figure(figsize=(16/2.56, 10/2.56))
    
    start, end = args.xrange
    bin_edges = np.arange(start, end + args.binstep, args.binstep)
    
    # Compute weights so that histogram counts are expressed in percentages.
    weights1 = np.ones_like(data1) / len(data1) * 100
    weights2 = np.ones_like(data2) / len(data2) * 100
    weights3 = np.ones_like(data3) / len(data3) * 100
    
    plt.hist(
        [data1, data2, data3],
        bins=bin_edges,
        weights=[weights1, weights2, weights3],
        label=args.legends,
        color=['blue', 'red', 'green'],
        edgecolor='black',
        linewidth=0.5,
        histtype='bar'
    )
    
    # Draw vertical lines at the mean values.
    plt.axvline(x=stats1['Mean'], color='blue', linestyle='--', linewidth=1)
    plt.axvline(x=stats2['Mean'], color='red', linestyle='--', linewidth=1)
    plt.axvline(x=stats3['Mean'], color='green', linestyle='--', linewidth=1)
    
    plt.xlabel('Latency (cycles)')
    plt.ylabel('Frequency (%)')
    plt.grid(which='both', linestyle='--', linewidth=0.5)
    plt.xlim(start, end)
    plt.ylim(0, args.ylim)
    
    # Create legend (handles might be duplicated due to vertical lines, so we use unique ones).
    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    plt.legend(unique.values(), unique.keys(), loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)

    legend = plt.legend(unique.values(), unique.keys(), loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    
    frame = legend.get_frame()
    frame.set_edgecolor('black')
    
    # Display computed statistics on the right side.
    stats_text = (
        f"{args.legends[0]}:\nMean: {stats1['Mean']:.2f}\nStd: {stats1['Std']:.2f}\nMax: {stats1['Max']:.2f}\nMin: {stats1['Min']:.2f}\n\n"
        f"{args.legends[1]}:\nMean: {stats2['Mean']:.2f}\nStd: {stats2['Std']:.2f}\nMax: {stats2['Max']:.2f}\nMin: {stats2['Min']:.2f}\n\n"
        f"{args.legends[2]}:\nMean: {stats3['Mean']:.2f}\nStd: {stats3['Std']:.2f}\nMax: {stats3['Max']:.2f}\nMin: {stats3['Min']:.2f}"
    )
    plt.gca().text(1.02, 0.5, stats_text, transform=plt.gca().transAxes,
                   fontsize=14, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
    
    plt.tight_layout()
    plt.savefig(args.plot_filename, bbox_inches='tight')
    print(f'Histogram saved as {args.plot_filename}')
    
if __name__ == "__main__":
    main()
