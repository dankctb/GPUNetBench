#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import matplotlib.ticker as ticker
import matplotlib.colors as mcolors

def main():
    parser = argparse.ArgumentParser(
        description="Generate a 2D colormap of bandwidth with corresponding {CTA ; warp} from a log file"
    )
    parser.add_argument("input_file",
                        help="Path to the log file (e.g. results_L2.log or results_HBM.log)")
    parser.add_argument("-o", "--output", default="2d_colormap.png",
                        help="Output image filename (default: %(default)s)")
    args = parser.parse_args()

    # ----------------------------------------------------------------
    # 1) Read data: blank lines separate 'slices' (y), each number is one 'sm' (x)
    # ----------------------------------------------------------------
    x, y, z = [], [], []
    warps_per_cta = 0
    cta_per_sm = 0

    with open(args.input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                warps_per_cta += 1
                cta_per_sm = 0
            else:
                bw = float(line)
                x.append(cta_per_sm)
                y.append(warps_per_cta)
                z.append(bw)
                cta_per_sm += 1

    x = np.array(x) # number of CTAs per SM
    y = np.array(y) # number of warps per CTA
    z = np.array(z) # bandwidth

    # ----------------------------------------------------------------
    # 2) Build a regular grid from min→max of x and y
    # ----------------------------------------------------------------
    xi = np.arange(x.min(), x.max() + 1)
    yi = np.arange(y.min(), y.max() + 1)
    xi_grid, yi_grid = np.meshgrid(xi, yi)

    zi = griddata((x, y), z, (xi_grid, yi_grid), method='cubic')

    # ----------------------------------------------------------------
    # 3) Auto color‐scale bounds
    # ----------------------------------------------------------------
    vmin = np.nanmin(zi)
    vmax = np.nanmax(zi)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)

    # ----------------------------------------------------------------
    # 4) Plot
    # ----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(24/2.54, 16/2.54))
    im = ax.imshow(
        zi,
        
        origin='lower',
        interpolation='nearest',
        aspect='auto',
        extent=(xi.min(), xi.max(), yi.min(), yi.max()),
        cmap='viridis',
        norm=norm
    )

    ax.set_xlabel('CTA per SM')
    ax.set_ylabel('Warp per CTA')

    # Tick every 4 units (you can adjust as desired)
    xticks = np.arange(xi.min(), xi.max() + 1, 4)
    yticks = np.arange(yi.min(), yi.max() + 1, 4)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)

    # Grid lines on both major and minor ticks
    ax.grid(which='major', color='black', linestyle='-', linewidth=0.7)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(4))
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.3)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, label='Bandwidth (GB/s)')
    # 6 evenly spaced ticks from vmin to vmax
    cbar.set_ticks(np.linspace(vmin, vmax, 6))

    plt.tight_layout()
    plt.savefig(args.output, dpi=600, bbox_inches='tight')

if __name__ == "__main__":
    main()
