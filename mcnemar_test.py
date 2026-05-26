import numpy as np
import csv
from itertools import combinations
from argparse import ArgumentParser
from pathlib import Path
from datasets import load_dataset
from statsmodels.stats.contingency_tables import mcnemar
import torch
from torch.utils.data import DataLoader

from halo_legibility.loader import HaLOPixtralFeatures
from halo_legibility.model import load_model
from halo_legibility.utils import parse_model_name, discover_models


def get_predictions(model, test_data, feat_dir, device, batch_size=64):
    """Returns binary array: 1 if model prediction correct, 0 otherwise."""
    dataset = HaLOPixtralFeatures(test_data, feat_dir)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    correct = []
    with torch.no_grad():
        for features, labels in loader:
            preds = model(features.to(device)).squeeze()
            predicted = (preds >= 0.5).long().cpu()
            labels = labels.long()
            correct.extend((predicted == labels).tolist())

    return np.array(correct, dtype=bool)


def main():
    parser = ArgumentParser(
        prog="McNemar Test",
        description="Pairwise McNemar's test between models on the test set",
    )
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--models-dir', default='models')
    parser.add_argument('-o', '--out-directory', default='results/ml-models')
    args = parser.parse_args()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device")

    dataset = load_dataset("MarcoLents/HaLO")
    test_data = dataset["test"]
    feat_dir = f"{args.data_dir}/features"

    model_names = discover_models(args.models_dir)
    print(f"Found models: {model_names}")

    # Get per-sample correctness for each model
    correctness = {}
    for name in model_names:
        print(f"Evaluating {name}...")
        arch, dropout = parse_model_name(name)
        model = load_model(name, arch, dropout, args.models_dir).to(device)
        correctness[name] = get_predictions(model, test_data, feat_dir, device)

    Path(args.out_directory).mkdir(parents=True, exist_ok=True)
    out_path = f"{args.out_directory}/mcnemar.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model_a", "model_b", "b", "c", "p_value"])
        writer.writeheader()

        for name_a, name_b in combinations(model_names, 2):
            a = correctness[name_a]
            b = correctness[name_b]

            # McNemar contingency table:
            # b = a correct, b wrong; c = a wrong, b correct
            b_count = int((a & ~b).sum())
            c_count = int((~a & b).sum())

            table = np.array([[int((a & b).sum()),  b_count],
                              [c_count, int((~a & ~b).sum())]])

            result = mcnemar(table, exact=True)

            print(f"  {name_a} vs {name_b}: b={b_count}, c={c_count}, p={result.pvalue:.4f}")
            writer.writerow({
                "model_a": name_a,
                "model_b": name_b,
                "b": b_count,
                "c": c_count,
                "p_value": result.pvalue,
            })

    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
