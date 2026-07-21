#!/usr/bin/env python3
"""Plot logical error rate vs. physical noise for the surface-code sweep
produced by run_surface_sweep.py, grouped by distance."""

import argparse

import matplotlib.pyplot as plt
import sinter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=str, default="surface_sweep_results.csv")
    parser.add_argument("--output", type=str, default="surface_sweep_plot.png")
    return parser


def main():
    args = build_parser().parse_args()
    stats = sinter.stats_from_csv_files(args.csv)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sinter.plot_error_rate(
        ax=ax,
        stats=stats,
        x_func=lambda stat: stat.json_metadata["p"],
        group_func=lambda stat: stat.json_metadata["d"],
        plot_args_func=lambda index, curve_id, group_stats: {"marker": "o"},
    )

    ax.loglog()
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel("physical error rate (p)")
    ax.set_ylabel("logical error rate")
    ax.set_title("Surface code memory experiment")
    ax.grid(which="both")
    ax.legend(title="distance", loc="best")

    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
