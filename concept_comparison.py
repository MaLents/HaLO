from huggingface_hub import snapshot_download
from halo_legibility.model import load_model
from scipy.stats import pearsonr, spearmanr
from argparse import ArgumentParser
import torch
import numpy as np
import glob
import os
import csv


CONCEPTS = ["isWrittenInPureCursive", "isStrokeThin", "containsTypo", "containsCorrection"]


def normalize(v):
    return v / np.linalg.norm(v)


def cosine_similarity(a, b):
    return np.dot(normalize(a), normalize(b))


def main():
    parser = ArgumentParser(
        prog="concept_comparison",
        description="Analyse concept vectors: cosine similarities, correlations, and direction removal",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("-o", "--out-directory", default="results/concepts")
    args = parser.parse_args()

    snapshot_download(
        repo_id="MarcoLents/HaLO",
        repo_type="dataset",
        allow_patterns="features/*/*.npy",
        local_dir=args.data_dir,
    )

    feat_dir = f"{args.data_dir}/features"
    concepts_dir = os.path.join(args.results_dir, "concepts")

    # --- Load concept vectors (output of find_concepts.py) ---
    concept_vectors = {}
    for concept in CONCEPTS:
        concept_vectors[concept] = np.load(os.path.join(concepts_dir, f"{concept}.npy"))
    concept_vectors["aspectRatio"] = np.load(os.path.join(concepts_dir, "aspectRatio.npy"))

    # --- Load m0 model and extract legibility weight vector ---
    model = load_model("m0", "m0", 0.25, args.models_dir)
    legibility_weights = model.classifier.fcs[0].weight.detach().squeeze().numpy()

    # --- Cosine similarities ---
    all_vectors = {"legibility": legibility_weights, **concept_vectors}
    names = list(all_vectors.keys())

    print("Cosine similarities:")
    cosine_rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            sim = cosine_similarity(all_vectors[a], all_vectors[b])
            print(f"  {a} vs {b}: {sim:.4f}")
            cosine_rows.append({"vector_a": a, "vector_b": b, "cosine_similarity": sim})

    cosine_path = os.path.join(args.out_directory, "cosine_similarities.csv")
    with open(cosine_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["vector_a", "vector_b", "cosine_similarity"])
        writer.writeheader()
        writer.writerows(cosine_rows)
    print(f"Saved {cosine_path}")

    # --- Load all features and compute model scores ---
    feature_files = sorted(glob.glob(os.path.join(feat_dir, "*/*.npy")))
    X_all = np.array([np.load(f) for f in feature_files])
    print(f"\nLoaded {X_all.shape[0]} feature vectors")

    model.eval()
    with torch.no_grad():
        scores = np.array([
            model.classifier(torch.tensor(feat).unsqueeze(0)).item()
            for feat in X_all
        ])

    # --- Pearson & Spearman correlations ---
    print("\nCorrelations (concept projection vs. model score):")
    corr_rows = []
    for name, vec in concept_vectors.items():
        proj = X_all @ normalize(vec)
        r_pearson, p_pearson = pearsonr(proj, scores)
        r_spearman, p_spearman = spearmanr(proj, scores)
        print(f"  {name}: Pearson r={r_pearson:.4f} (p={p_pearson:.2e}), Spearman r={r_spearman:.4f}")
        corr_rows.append({
            "concept": name,
            "pearson_r": r_pearson, "pearson_p": p_pearson,
            "spearman_r": r_spearman, "spearman_p": p_spearman,
        })

    corr_path = os.path.join(args.out_directory, "correlations.csv")
    with open(corr_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept", "pearson_r", "pearson_p", "spearman_r", "spearman_p"])
        writer.writeheader()
        writer.writerows(corr_rows)
    print(f"Saved {corr_path}")

    # --- Direction removal analysis ---
    leg_norm = normalize(legibility_weights)

    print("\nDirection removal (residual correlation with legibility after removing concept):")
    removal_rows = []
    for name, vec in concept_vectors.items():
        vec_norm = normalize(vec)
        proj = (X_all @ vec_norm)[:, None]
        X_resid = X_all - proj * vec_norm
        resid_proj = X_resid @ leg_norm
        r_pearson, _ = pearsonr(resid_proj, scores)
        r_spearman, _ = spearmanr(resid_proj, scores)
        print(f"  without {name}: Pearson r={r_pearson:.4f}, Spearman r={r_spearman:.4f}")
        removal_rows.append({
            "removed_concept": name,
            "residual_pearson_r": r_pearson,
            "residual_spearman_r": r_spearman,
        })

    removal_path = os.path.join(args.out_directory, "direction_removal.csv")
    with open(removal_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["removed_concept", "residual_pearson_r", "residual_spearman_r"])
        writer.writeheader()
        writer.writerows(removal_rows)
    print(f"Saved {removal_path}")


if __name__ == "__main__":
    main()
