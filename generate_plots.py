import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from pathlib import Path
from argparse import ArgumentParser


def plot_bootstrap_kde(results_dir, metric, xlabel, out_path):
    """Figure: KDE of bootstrap rank correlations (tau or rho)."""
    main_dir = Path(results_dir) / "main-study"

    MAIN_STUDY_LABELS = {
        "train": "Train",
        "validation": "Val",
        "test": "Test",
    }

    fig, ax = plt.subplots(figsize=(10, 6))

    reds = plt.cm.Reds(np.linspace(0.3, 0.9, len(MAIN_STUDY_LABELS)))

    # Main study splits
    for i, (key, label) in enumerate(MAIN_STUDY_LABELS.items()):
        data = np.load(main_dir / key / f"{metric}.npy")
        sns.kdeplot(data, label=label, fill=True, alpha=0.1, linewidth=2, color=reds[i], ax=ax)

    ax.axvline(x=1.0, color='black', linestyle='--', linewidth=1)

    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel("Density", fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_yticks([])
    ax.legend(fontsize=12, loc="upper right")

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
