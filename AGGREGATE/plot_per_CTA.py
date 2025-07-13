#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-CTA bar plots of bandwidth vs warp (or SM index ???) from a log file"
    )
    parser.add_argument("input_file",
                        help="Path to the log file (e.g. results_L2.log or results_HBM.log)")
    parser.add_argument("-d", "--output-dir", default=".",
                        help="Directory to save the per-slice plots (default: current dir)")
    parser.add_argument("-s", "--suffix", default="",
                        help="Suffix to append to each output filename (e.g. 'L2' or 'HBM')")
    args = parser.parse_args()

    # ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # ----------------------------------------------------------------
    # 1) Read data: blank lines separate 'slices' (y), each number is one 'sm' (x)
    # ----------------------------------------------------------------
    x, y, z = [], [], []
    slice_idx = 0
    sm_idx = 0

    with open(args.input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: # blank line detected => new CTA
                slice_idx += 1
                sm_idx = 0
            else: # new warp (or SM index ???)
                bw = float(line)
                x.append(sm_idx)
                y.append(slice_idx)
                z.append(bw)
                sm_idx += 1

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # ----------------------------------------------------------------
    # 2) Compute automatic y-axis bounds (0 → max*1.05)
    # ----------------------------------------------------------------
    y_min = 0
    y_max = z.max() * 1.05

    # ----------------------------------------------------------------
    # 3) Loop over each slice and make a bar plot
    # ----------------------------------------------------------------
    unique_slices = np.unique(y)
    for slice_num in unique_slices:
        mask = (y == slice_num)
        slice_x = x[mask]
        slice_z = z[mask]

        fig, ax = plt.subplots(figsize=(24/2.54, 16/2.54))

        ax.bar(slice_x, slice_z, color='grey')
        ax.set_ylim(y_min, y_max)

        # X‑ticks: one bar per SM index, label as 1-based
        ax.set_xticks(slice_x)
        ax.set_xticklabels((slice_x + 1).astype(int), rotation=90)

        # Y‑ticks: automatic nice integers
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

        ax.set_xlabel('SM index (CTA per SM)')
        ax.set_ylabel('Bandwidth (GB/s)')
        ax.set_title(f'Slice {slice_num}')

        plt.tight_layout()

        # Build filename with optional suffix
        suffix = f"_{args.suffix}" if args.suffix else ""
        out_file = os.path.join(
            args.output_dir,
            f"slice_{slice_num:02d}{suffix}.png"
        )

        plt.savefig(out_file, dpi=600, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(unique_slices)} plots to '{args.output_dir}'")

if __name__ == "__main__":
    main()
