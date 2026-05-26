import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def plot_ranking(results_dir, out_path):
    """Figure: Ranking visualization with mean strength ± bootstrap SD."""
    splits = [
        ("train", "Train Partition"),
        ("validation", "Validation Partition"),
        ("test", "Test Partition"),
    ]

    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, height_ratios=[2, 1], hspace=0.35, wspace=0.3)

    for idx, (split, title) in enumerate(splits):
        d = Path(results_dir) / "main-study" / split
        theta = np.load(d / "theta.npy")
        theta_std = np.load(d / "theta-std.npy")

        order = np.argsort(theta)
        theta_sorted = theta[order]
        std_sorted = theta_std[order]
        n = len(theta_sorted)

        if idx == 0:
            ax = fig.add_subplot(gs[0, :])
        else:
            ax = fig.add_subplot(gs[1, idx - 1])

        ax.fill_between(
            range(n),
            theta_sorted - std_sorted,
            theta_sorted + std_sorted,
            color="blue", alpha=0.2,
        )
        ax.plot(range(n), theta_sorted, color="blue", marker="o", markersize=1.5, linewidth=0.5)
        ax.set_xlabel(f"Samples (N={n}; ordered ascending by strength)", fontsize=10)
        ax.set_ylabel("Strength value", fontsize=10)
        ax.set_title(title, fontsize=12)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_bootstrap_kde(results_dir, metric, xlabel, out_path):
    """Figure: KDE of bootstrap rank correlations (tau or rho)."""
    main_dir = Path(results_dir) / "main-study"
    pre_dir = Path(results_dir) / "pre-study"

    PRE_STUDY_LABELS = {
        "annotator11": "Annotator$1_1$",
        "annotator12": "Annotator$1_2$",
        "annotator2": "Annotator$2$",
        "annotator3": "Annotator$3$",
    }
    MAIN_STUDY_LABELS = {
        "train": "Train",
        "validation": "Val",
        "test": "Test",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    blues = plt.cm.Blues(np.linspace(0.3, 0.9, len(PRE_STUDY_LABELS)))
    reds = plt.cm.Reds(np.linspace(0.3, 0.9, len(MAIN_STUDY_LABELS)))

    # Pre-study annotators (ordered to match paper legend)
    for i, (key, label) in enumerate(PRE_STUDY_LABELS.items()):
        data = np.load(pre_dir / key / f"{metric}.npy")
        sns.kdeplot(data, label=label, fill=True, alpha=0.1, linewidth=2, color=blues[i], ax=ax)

    # Main study splits
    for i, (key, label) in enumerate(MAIN_STUDY_LABELS.items()):
        data = np.load(main_dir / key / f"{metric}.npy")
        sns.kdeplot(data, label=label, fill=True, alpha=0.1, linewidth=2, color=reds[i], ax=ax)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_yticks([])
    ax.legend(fontsize=12)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = ArgumentParser(
        prog="generate_plots",
        description="Generate paper figures from analysis results",
    )
    parser.add_argument("-r", "--results-dir", default="results")
    parser.add_argument("-o", "--out-directory", default="figures")
    args = parser.parse_args()

    Path(args.out_directory).mkdir(parents=True, exist_ok=True)

    plot_ranking(
        args.results_dir,
        f"{args.out_directory}/ranking.png",
    )

    plot_bootstrap_kde(
        args.results_dir,
        metric="tau",
        xlabel="Kendall rank correlation coefficient",
        out_path=f"{args.out_directory}/bootstrap_tau.png",
    )

    plot_bootstrap_kde(
        args.results_dir,
        metric="rho",
        xlabel="Spearman rank correlation coefficient",
        out_path=f"{args.out_directory}/bootstrap_rho.png",
    )


if __name__ == "__main__":
    main()
