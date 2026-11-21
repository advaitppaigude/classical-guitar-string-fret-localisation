"""Plot agreement, clipping, and selection criteria from the B/F sweep CSV."""

import argparse
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, PowerNorm
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator


RESULTS = Path(__file__).resolve().parents[1] / "results"
DEFAULT_CSV = RESULTS / "numerical_exploration/bit_fraction_sweep.csv"
UNTESTED_COLOR = "#e3e7ec"
CLIP_GROUPS = (
    ("later_feature", "Later-session features"),
    ("original_feature", "Original-set features"),
    ("weight", "Weights"),
    ("bias", "Biases"),
)


def heatmap(ax, values, title, cmap, norm):
    grid = values.unstack("B")
    image = ax.imshow(
        grid.to_numpy(),
        origin="lower",
        interpolation="nearest",
        cmap=plt.get_cmap(cmap).with_extremes(bad=UNTESTED_COLOR),
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xticks(range(len(grid.columns)), grid.columns)
    ax.set_yticks(range(len(grid.index)), grid.index)
    ax.tick_params(length=0, labelsize=8)
    return image


def plot_agreement(data, path):
    later = 100 * data["later_matches"] / data["later_total"]
    original = 100 * data["original_matches"] / data["original_total"]
    lower = min(later.min(), original.min(), 99.0)
    norm = Normalize(lower, 100)
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), sharex=True, sharey=True, layout="constrained")
    agreement = heatmap(axes[0], later, "Later recordings", "viridis", norm)
    heatmap(axes[1], original, "Original recordings", "viridis", norm)
    fig.colorbar(agreement, ax=axes, label="Agreement with floating point (%)", shrink=0.82)
    fig.suptitle("Fixed-point agreement")
    fig.supxlabel("Total bits B")
    fig.supylabel("Fractional bits F")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_clipping(data, path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 10.5), sharex=True, sharey=True, layout="constrained")
    for column, (group, title) in enumerate(CLIP_GROUPS):
        for row, (field, label) in enumerate((
            ("clips", "Clipped values"),
            ("max_clip_excess", "Max excess beyond range"),
        )):
            values = data[f"{group}_{field}"]
            maximum = values.max()
            norm = PowerNorm(gamma=0.45, vmin=0, vmax=maximum if maximum > 0 else 1)
            panel_title = f"{title}\nAll zero" if maximum == 0 else title
            ax = axes[row, column]
            image = heatmap(ax, values, panel_title, "magma", norm)
            ticks = [0] if maximum == 0 else MaxNLocator(nbins=5, integer=field == "clips")
            fig.colorbar(image, ax=ax, label=label, ticks=ticks, shrink=0.85)
    fig.suptitle("Clipping counts and maximum excess")
    fig.supxlabel("Total bits B")
    fig.supylabel("Fractional bits F")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_selection(data, path):
    full_agreement = (
        data["original_matches"].eq(data["original_total"])
        & data["later_matches"].eq(data["later_total"])
    )
    any_clipping = data[[f"{group}_clips" for group, _ in CLIP_GROUPS]].gt(0).any(axis=1)
    categories = pd.Series(0, index=data.index)
    categories.loc[full_agreement] = 1
    categories.loc[full_agreement & ~any_clipping] = 2

    colors = ("#515b67", "#edb440", "#63b587")
    cmap = ListedColormap(colors)
    norm = BoundaryNorm((-0.5, 0.5, 1.5, 2.5), cmap.N)
    fig, ax = plt.subplots(figsize=(8.5, 7.1), layout="constrained")
    heatmap(ax, categories, "Agreement for the original and later recordings", cmap, norm)
    grid = categories.unstack("B")

    shortlist = {
        data.loc[eligible, "rank"].idxmin()
        for eligible in (full_agreement, full_agreement & ~any_clipping)
        if eligible.any()
    }
    for frac_bits, bits in sorted(shortlist):
        x, y = grid.columns.get_loc(bits), grid.index.get_loc(frac_bits)
        ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor="#18242d", linewidth=1.4))
        ax.text(x, y, f"B{bits}_F{frac_bits}", ha="center", va="center", fontsize=6.3, fontweight="bold")

    ax.legend(
        handles=[
            Patch(facecolor=colors[2], label="Full agreement, zero clipping"),
            Patch(facecolor=colors[1], label="Full agreement, some clipping"),
            Patch(facecolor=colors[0], label="A prediction differs from float"),
            Patch(facecolor=UNTESTED_COLOR, label="Untested combination"),
        ],
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=False,
    )
    fig.suptitle("Fixed-point format selection")
    ax.set_xlabel("Total bits B")
    ax.set_ylabel("Fractional bits F")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-dir", type=Path, default=RESULTS / "plots")
    args = parser.parse_args()
    data = pd.read_csv(args.csv).set_index(["F", "B"]).sort_index()
    if not data.index.is_unique:
        parser.error("CSV has duplicate (B, F) combinations")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    agreement = args.output_dir / "bit_fraction_agreement.png"
    clipping = args.output_dir / "bit_fraction_clipping.png"
    selection = args.output_dir / "bit_fraction_selection.png"
    plot_agreement(data, agreement)
    plot_clipping(data, clipping)
    plot_selection(data, selection)
    print(agreement)
    print(clipping)
    print(selection)


if __name__ == "__main__":
    main()
