import itertools
import numpy as np
import pandas as pd
import krippendorff
from pathlib import Path
from datasets import load_dataset
from argparse import ArgumentParser
import torch

from halo_legibility.model import load_model
from halo_legibility.loader import score_images
from halo_legibility.utils import parse_model_name, discover_models


def fractions_to_counts(fractions, n):
    """Convert rating fractions to integer counts summing to n (largest remainder)."""
    scale = sorted(fractions)
    raw = {r: fractions[r] * n for r in scale}
    counts = {r: int(v) for r, v in raw.items()}
    remainder = n - sum(counts.values())
    for r in sorted(scale, key=lambda r: raw[r] - counts[r], reverse=True)[:remainder]:
        counts[r] += 1
    return counts


def map_scores_to_ratings(scores_dict, annotator_ratings, fractions=None):
    """
    Map model scores to discrete ratings.

    If fractions is given ({rating: fraction}), use those fixed fractions.
    Otherwise match the annotator's own distribution.

    scores_dict: {sample_id: score}  (higher = better)
    annotator_ratings: {sample_id: rating}  (lower = better)
    Returns: {sample_id: predicted_rating}
    """
    common_ids = [id_ for id_ in annotator_ratings if id_ in scores_dict]
    if fractions is not None:
        counts = fractions_to_counts(fractions, len(common_ids))
        scale = sorted(fractions)
    else:
        ratings = [annotator_ratings[id_] for id_ in common_ids]
        scale = sorted(set(ratings))
        counts = {r: ratings.count(r) for r in scale}

    sorted_ids = sorted(common_ids, key=lambda id_: scores_dict[id_], reverse=True)

    result = {}
    pos = 0
    for r in scale:
        for _ in range(counts[r]):
            result[sorted_ids[pos]] = r
            pos += 1
    return result


def main():
    parser = ArgumentParser(
        prog="Absolute Analysis",
        description="Performs the analysis of the absolute annotations from the paper",
    )

    parser.add_argument('--no-ml', action='store_true')
    parser.add_argument('--models-dir', default="models")
    parser.add_argument('--data-dir', default="data")

    args = parser.parse_args()

    # --- Pre-study analysis (Table 3) ---
    df_pre = pd.DataFrame(load_dataset("MarcoLents/HaLO", "absolute_pre_study")["train"])
    annotator_cols_pre = ["annotator41", "annotator42", "annotator5", "annotator6", "annotator7"]

    all_ratings = df_pre[annotator_cols_pre].apply(pd.to_numeric, errors="raise").to_numpy().T

    print("Krippendorff's alpha (pre-study)")
    Path("results/pre-study").mkdir(parents=True, exist_ok=True)
    with open("results/pre-study/alphas.csv", "w") as alpha_file:
        alpha_file.write("Annotators, alpha")

        alpha_all = krippendorff.alpha(
            reliability_data=all_ratings,
            level_of_measurement="ordinal",
        )

        alpha_file.write(f"\nAll, {alpha_all}")
        print(f"  All: {alpha_all:.4f}")

        for col_a, col_b in itertools.combinations(annotator_cols_pre, 2):
            pair_ratings = df_pre[[col_a, col_b]].apply(pd.to_numeric, errors="raise").to_numpy().T
            alpha_pair = krippendorff.alpha(
                reliability_data=pair_ratings,
                level_of_measurement="ordinal",
            )

            alpha_file.write(f"\n{col_a}-{col_b}, {alpha_pair}")
            print(f"  {col_a} vs {col_b}: {alpha_pair:.4f}")

    # --- Test-set analysis (Table 5) ---
    df_test = pd.DataFrame(load_dataset("MarcoLents/HaLO", "absolute_test")["train"])
    annotator_cols_test = ["annotator8", "annotator9", "annotator10", "annotator11"]

    sample_ids = df_test["sampleId"].tolist()

    # Build {annotator: {sample_id: rating}} for non-NaN entries
    ann_ratings_map = {}
    for col in annotator_cols_test:
        ann_ratings_map[col] = {
            row["sampleId"]: int(row[col])
            for _, row in df_test.iterrows()
            if pd.notna(row[col])
        }

    ann_counts = {col: len(v) for col, v in ann_ratings_map.items()}

    # Reliability matrix
    rating_matrix = np.full((len(annotator_cols_test), len(sample_ids)), np.nan)
    for i, col in enumerate(annotator_cols_test):
        for j, sid in enumerate(sample_ids):
            if sid in ann_ratings_map[col]:
                rating_matrix[i, j] = ann_ratings_map[col][sid]

    # Pairwise annotator alphas
    print("\nKrippendorff's alpha (test set, Table 5)")
    Path("results/main-study").mkdir(parents=True, exist_ok=True)
    with open("results/main-study/alphas_test.csv", "w") as alpha_file:
        alpha_file.write("Annotator1, Annotator2, alpha")
        for i, col_a in enumerate(annotator_cols_test):
            for j, col_b in enumerate(annotator_cols_test):
                if j <= i:
                    continue
                pair = rating_matrix[[i, j], :]
                mask = ~np.isnan(pair).any(axis=0)
                alpha = krippendorff.alpha(
                    reliability_data=pair[:, mask],
                    level_of_measurement="ordinal",
                )
                alpha_file.write(f"\n{col_a}, {col_b}, {alpha}")
                print(f"  {col_a} vs {col_b}: {alpha:.4f} (n={int(mask.sum())})")

    if args.no_ml:
        print("Skipping analysis of machine-learning models")
        return

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    feat_dir = f"{args.data_dir}/features"
    id_to_path = dict(zip(df_test["sampleId"], df_test["samplePath"]))

    # Compute average rating fractions across annotators
    scale = list(range(5))
    ann_fracs = []
    for col in annotator_cols_test:
        ratings = list(ann_ratings_map[col].values())
        if ratings:
            ann_fracs.append({r: ratings.count(r) / len(ratings) for r in scale})
    avg_fractions = {r: np.mean([f[r] for f in ann_fracs]) for r in scale}

    # Score test images with the linear probe (m0)
    m0_name = next(n for n in discover_models(args.models_dir) if n.startswith("m0_"))
    arch, dropout = parse_model_name(m0_name)
    model = load_model(m0_name, arch, dropout, args.models_dir).to(device)
    scores_dict = score_images(model, feat_dir, id_to_path, device)

    print("\nKrippendorff's alpha: annotators vs linear probe (test set)")
    with open("results/main-study/alphas_test_ml.csv", "w") as alpha_file:
        alpha_file.write("Annotator, alpha")
        for col in annotator_cols_test:
            predicted = map_scores_to_ratings(scores_dict, ann_ratings_map[col], fractions=avg_fractions)
            common_ids = sorted(predicted.keys())
            pred_array = np.array([predicted[id_] for id_ in common_ids])
            true_array = np.array([ann_ratings_map[col][id_] for id_ in common_ids])

            alpha = krippendorff.alpha(
                reliability_data=np.array([pred_array, true_array]),
                level_of_measurement="ordinal",
            )
            alpha_file.write(f"\n{col}, {alpha}")
            print(f"  {col} vs linear probe: {alpha:.4f}")


if __name__ == "__main__":
    main()
