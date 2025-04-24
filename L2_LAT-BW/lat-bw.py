#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

# Set the global font to be DejaVu Sans, size 14 (all text will be this size)
plt.rcParams['font.size'] = 16
plt.rcParams['font.family'] = 'DejaVu Sans'

# Set the font size of the axes to be 12
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16


def read_values(filepath):
    """
    Reads a file with one numeric value per line and returns a numpy array.
    """
    try:
        return np.loadtxt(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(
        description="Plot L2 Throughput (GB/s) vs. average latency or kernel execution time for multiple data series."
    )
    parser.add_argument("--bw_files", nargs="+", required=True,
                        help="List of bandwidth files (each file contains one measurement per injection rate, in GB/s)")
    parser.add_argument("--lat_files", nargs="+", required=True,
                        help=("List of latency (or kernel execution time) files (each file contains one measurement per injection rate). "
                              "If yaxis is 'latency', values are in clock cycles; if 'exec_time', values are in µs."))
    parser.add_argument("--legends", nargs="+", required=True,
                        help="Legend labels for each data series (one label per pair of files)")
    parser.add_argument("--yaxis", choices=["latency", "exec_time"], default="latency",
                        help=("Choose which metric to plot on the y-axis: "
                              "'latency' (in clock cycles) or 'exec_time' (in microseconds)"))
    parser.add_argument("--output", default="plot.png",
                        help="Output filename for the plot (default: plot.png)")
    args = parser.parse_args()

    # Ensure the number of files and legend labels match.
    if not (len(args.bw_files) == len(args.lat_files) == len(args.legends)):
        parser.error("The number of bandwidth files, latency files, and legend labels must be equal.")

    series_data = []
    for bw_file, lat_file, legend in zip(args.bw_files, args.lat_files, args.legends):
        bw_vals = read_values(Path(bw_file))
        lat_vals = read_values(Path(lat_file))
        if bw_vals is None or lat_vals is None:
            print(f"Skipping series '{legend}' due to read error.")
            continue
        if len(bw_vals) != len(lat_vals):
            print(f"Warning: Number of measurements in {bw_file} and {lat_file} do not match for series '{legend}'.")
        series_data.append({
            "bw": bw_vals,
            "lat": lat_vals,
            "label": legend
        })

    if not series_data:
        print("No valid series data to plot.")
        return

    # Create the plot.
    fig, ax = plt.subplots(figsize=(16/2.56, 10/2.56))

    # Compute overall x- and y-limits from all series.
    all_bw = np.concatenate([s["bw"] for s in series_data])
    all_y = np.concatenate([s["lat"] for s in series_data])
    x_min, x_max = np.min(all_bw), np.max(all_bw)
    y_min, y_max = np.min(all_y), np.max(all_y)
    # Add a small padding.
    x_padding = 0.1 * (np.log2(x_max) - np.log2(x_min))
    y_padding = 0.05 * (y_max - y_min)

    for s in series_data:
        # x-values are taken directly from the bandwidth file (GB/s).
        # y-values are from the latency (or exec_time) file.
        ax.plot(s["bw"], s["lat"], marker='o', linestyle='-', label=s["label"])

    ax.set_xlabel("L2 Throughput (GB/s)")
    if args.yaxis == "latency":
        ax.set_ylabel("Average Latency (clock cycles)")
    else:
        ax.set_ylabel("Kernel Execution Time (µs)")

    ax.set_xscale("log", base=2)
    ax.set_xlim(2**(np.log2(x_min) - x_padding), 2**(np.log2(x_max) + x_padding))
    ax.set_ylim(y_min - y_padding, y_max + y_padding)

    # Set x-axis ticks at powers of 2.
    ticks = [2**i for i in range(int(np.floor(np.log2(x_min))), int(np.ceil(np.log2(x_max))) + 1)]
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FixedFormatter([str(tick) for tick in ticks]))

    ax.grid(which="both", linestyle="--", linewidth=0.5)
    legend = ax.legend(unique.values(), unique.keys(), loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3)
    
    frame = legend.get_frame()
    frame.set_edgecolor('black')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved as {args.output}")

if __name__ == "__main__":
    main()
