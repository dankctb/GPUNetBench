#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-SM bar plots of bandwidth vs slice index from a log file"
    )
    parser.add_argument("input_file",
                        help="Path to the log file (e.g. results_L2.log or results_HBM.log)")
    parser.add_argument("-d", "--output-dir", default=".",
                        help="Directory to save the per-SM plots (default: current dir)")
    parser.add_argument("-s", "--suffix", default="",
                        help="Suffix to append to each output filename (e.g. 'L2' or 'HBM')")
    args = parser.parse_args()

    # Ensure output directory exists
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
            if not line:
                slice_idx += 1
                sm_idx = 0
            else:
                bw = float(line)
                x.append(sm_idx)
                y.append(slice_idx)
                z.append(bw)
                sm_idx += 1

    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # ----------------------------------------------------------------
    # 2) Compute auto y-axis bounds (0 → max*1.05)
    # ----------------------------------------------------------------
    y_min = 0
    y_max = z.max() * 1.05

    # ----------------------------------------------------------------
    # 3) Loop over each SM and make a bar plot
    # ----------------------------------------------------------------
    unique_sms = np.unique(x)
    for sm_num in unique_sms:
        mask = (x == sm_num)
        sm_y = y[mask]
        sm_z = z[mask]

        fig, ax = plt.subplots(figsize=(24/2.54, 16/2.54))
        ax.bar(sm_y, sm_z, color='grey')
        ax.set_ylim(y_min, y_max)

        # X‑ticks: one bar per slice index, label as 1-based
        ax.set_xticks(sm_y)
        ax.set_xticklabels((sm_y + 1).astype(int), rotation=90)

        # Y‑ticks: nice integer locator
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))

        ax.set_xlabel('CTA')
        ax.set_ylabel('Bandwidth (GB/s)')
        ax.set_title(f'WARP {sm_num}')

        plt.tight_layout()

        # Build filename with optional suffix
        suffix = f"_{args.suffix}" if args.suffix else ""
        out_file = os.path.join(
            args.output_dir,
            f"sm_{sm_num:02d}{suffix}.png"
        )

        plt.savefig(out_file, dpi=600, bbox_inches='tight')
        plt.close(fig)

    print(f"Saved {len(unique_sms)} plots to '{args.output_dir}'")

if __name__ == "__main__":
    main()
