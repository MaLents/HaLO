import itertools
from math import floor
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


def get_fractions(ann_table, annotator_cols):
    # Compute average rating fractions across annotators for val set
    scale = list(range(5))
    ann_counts = {i: 0 for i in scale}
    total_counts = 0
    for col in annotator_cols:
        ratings = list(ann_table[col].values())
        if ratings:
            for r in scale:
                ann_counts[r] += ratings.count(r)
            total_counts += len(ratings)
    fractions = {r: ann_counts[r] / total_counts for r in scale}
    
    return fractions

def get_thresholds(scores, fractions):
    sorted_scores = sorted(scores.values(), reverse=True)

    thresholds = []

    cumm = 0.0
    for r in range(4):
        cumm += fractions[r]
        idx = floor(cumm * len(sorted_scores))
        thresholds.append((sorted_scores[idx] + sorted_scores[idx+1]) / 2)

    return thresholds


def map_scores_to_ratings(scores_dict, thresholds):
    """
    Map model scores to discrete ratings.

    If fractions is given ({rating: fraction}), use those fixed fractions.
    Otherwise match the annotator's own distribution.

    scores_dict: {sample_id: score}  (higher = better)
    annotator_ratings: {sample_id: rating}  (lower = better)
    Returns: {sample_id: predicted_rating}
    """
    out = {}

    for sample_id, score in scores_dict.items():
        out[sample_id] = len([t for t in thresholds if t > score])

    return out


def main():
    parser = ArgumentParser(
        prog="Absolute Analysis",
        description="Performs the analysis of the absolute annotations from the paper",
    )

    parser.add_argument('--no-ml', action='store_true')
    parser.add_argument('--models-dir', default="models")
    parser.add_argument('--data-dir', default="data")

    args = parser.parse_args()

    # --- Absolute annotation analysis (Table 3) ---
    df_test = pd.DataFrame(load_dataset("MarcoLents/HaLO", "absolute")["test"])
    annotator_cols_main = ["A1", "A2", "A3", "A4"]

    # --- Val-set analysis ---
    df_val = pd.DataFrame(load_dataset("MarcoLents/HaLO", "absolute")["validation"])

    sample_ids = df_test["sampleId"].tolist()
    val_ids = df_val["sampleId"].tolist()

    # Build {annotator: {sample_id: rating}} for non-NaN entries
    ann_ratings_map_test = {}
    for col in annotator_cols_main:
        ann_ratings_map_test[col] = {
            row["sampleId"]: int(row[col])
            for _, row in df_test.iterrows()
            if pd.notna(row[col])
        }

    # Build {annotator: {sample_id: rating}} for non-NaN entries
    ann_ratings_map_val = {}
    for col in annotator_cols_main:
        ann_ratings_map_val[col] = {
            row["sampleId"]: int(row[col])
            for _, row in df_val.iterrows()
            if pd.notna(row[col])
        }

    ann_counts = {col: len(v) for col, v in ann_ratings_map_test.items()}
    ann_counts_val = {col: len(v) for col, v in ann_ratings_map_val.items()}

    # Reliability matrix
    rating_matrix = np.full((len(annotator_cols_main), len(sample_ids)), np.nan)
    for i, col in enumerate(annotator_cols_main):
        for j, sid in enumerate(sample_ids):
            if sid in ann_ratings_map_test[col]:
                rating_matrix[i, j] = ann_ratings_map_test[col][sid]

    # Pairwise annotator alphas
    print("\nKrippendorff's alpha (test set, Table 3)")
    Path("results/main-study").mkdir(parents=True, exist_ok=True)
    with open("results/main-study/alphas_test.csv", "w") as alpha_file:
        alpha_file.write("Annotator1, Annotator2, alpha, N")
        for i, col_a in enumerate(annotator_cols_main):
            for j, col_b in enumerate(annotator_cols_main):
                if j <= i:
                    continue
                pair = rating_matrix[[i, j], :]
                mask = ~np.isnan(pair).any(axis=0)
                alpha = krippendorff.alpha(
                    reliability_data=pair[:, mask],
                    level_of_measurement="ordinal",
                )
                n = int(mask.sum())
                alpha_file.write(f"\n{col_a}, {col_b}, {alpha}, {n}")
                print(f"  {col_a} vs {col_b}: {alpha:.4f} (N={n})")

    if args.no_ml:
        print("Skipping analysis of machine-learning models")
        return

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    feat_dir = f"{args.data_dir}/features"
    id_to_path = dict(zip(df_test["sampleId"], df_test["samplePath"]))
    id_to_path_val = dict(zip(df_val["sampleId"], df_val["samplePath"]))

    val_fracs = get_fractions(ann_ratings_map_val, annotator_cols_main)

    # Score test images with the linear probe (m0)
    m0_name = next(n for n in discover_models(args.models_dir) if n.startswith("m0_"))
    arch, dropout = parse_model_name(m0_name)
    model = load_model(m0_name, arch, dropout, args.models_dir).to(device)
    scores_dict = score_images(model, feat_dir, id_to_path, device)

    val_scores_dict = score_images(model, feat_dir, id_to_path_val, device)

    val_thres = get_thresholds(val_scores_dict, val_fracs)

    print("\nKrippendorff's alpha: annotators vs linear probe (test set)")
    with open("results/main-study/alphas_test_ml.csv", "w") as alpha_file:
        alpha_file.write("Annotator, alpha, N")
        for col in annotator_cols_main:
            predicted = map_scores_to_ratings(scores_dict, val_thres)
            common_ids = list(filter(lambda k: k in ann_ratings_map_test[col].keys(), sorted(predicted.keys())))
            pred_array = np.array([predicted[id_] for id_ in common_ids])
            true_array = np.array([ann_ratings_map_test[col][id_] for id_ in common_ids])

            alpha = krippendorff.alpha(
                reliability_data=np.array([pred_array, true_array]),
                level_of_measurement="ordinal",
            )
            n = len(common_ids)
            alpha_file.write(f"\n{col}, {alpha}, {n}")
            print(f"  {col} vs linear probe: {alpha:.4f} (N={n})")


if __name__ == "__main__":
    main()
