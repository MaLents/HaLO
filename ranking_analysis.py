import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.special import expit
from scipy import stats
from datasets import load_dataset
from argparse import ArgumentParser
from pathlib import Path
from itertools import combinations
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from halo_legibility.rank import fit_model, eval_model
from halo_legibility.model import load_model
from halo_legibility.loader import HaLOPixtralFeatures, score_images
from halo_legibility.utils import parse_model_name, discover_models

def compare_models(model1, model2):
    rho = stats.spearmanr(model1.thetas, model2.thetas).statistic
    tau = stats.kendalltau(model1.thetas, model2.thetas).statistic

    return rho, tau

def bootstrap_analysis(df, samples: int):
    base_model = fit_model(df)
    base_auc, base_acc = eval_model(base_model, df)
    base_thetas, base_variances, base_ranks = base_model.characterize_distributions()

    thetas, variances, ranks = [], [], []
    rhos, taus = [], []
    aucs, accs = [], []

    for _ in tqdm(range(samples), desc="  resampling", leave=False):
        resample = df.sample(n=len(df), replace=True)
        model = fit_model(resample)

        t, v, r = model.characterize_distributions()
        thetas.append(t)
        variances.append(v)
        ranks.append(r)

        rho, tau = compare_models(base_model, model)
        rhos.append(rho)
        taus.append(tau)

        auc, acc = eval_model(model, resample)
        aucs.append(auc)
        accs.append(acc)

    theta_std    = np.std(thetas, axis=0)
    variance_std = np.std(variances, axis=0)
    rank_std     = np.std(ranks, axis=0)

    return (base_thetas, base_variances, base_ranks,
            theta_std, variance_std, rank_std,
            rhos, taus, base_auc, base_acc, aucs, accs)

def eval_model_ml(model, test_data, feat_dir, device, batch_size=64):
    dataset = HaLOPixtralFeatures(test_data, feat_dir)
    loader = DataLoader(dataset, batch_size=batch_size)

    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in loader:
            preds = model(features.to(device))
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels).astype(int)

    auc = roc_auc_score(all_labels, all_preds)
    acc = accuracy_score(all_labels, np.rint(all_preds))
    return auc, acc


def main():
    parser = ArgumentParser(
        prog="Ranking Analysis",
        description="Perform the ranking analyses on the BT model",
    )

    parser.add_argument('--resamples', type=int, default=1000)
    parser.add_argument('-o', '--out-directory', default="results")
    parser.add_argument('--no-ml', action='store_true')
    parser.add_argument('--models-dir', default="models")
    parser.add_argument('--data-dir', default="data")
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    np.random.seed(args.seed)

    main_annotations = load_dataset("MarcoLents/HaLO")

    print("Main study bootstrap analysis")
    Path(f"{args.out_directory}/main-study").mkdir(parents=True, exist_ok=True)
    with open(f"{args.out_directory}/main-study/performance.csv", "w") as main_file:
        main_file.write("split, ACC, sigma(ACC), AUC, sigma(AUC), <sigma(rank)>")
        for key, split in main_annotations.items():
            Path(f"{args.out_directory}/main-study/{key}").mkdir(parents=True, exist_ok=True)
            print(f"  Fitting [{key}] ({args.resamples} resamples)...", flush=True)

            (base_thetas, base_variances, base_ranks,
             theta_std, variance_std, rank_std,
             rhos, taus, base_auc, base_acc,
             aucs, accs) = bootstrap_analysis(pd.DataFrame(split), samples=args.resamples)

            d = f"{args.out_directory}/main-study/{key}"
            np.save(f"{d}/theta.npy", base_thetas)
            np.save(f"{d}/var.npy", base_variances)
            np.save(f"{d}/ranks.npy", base_ranks)
            np.save(f"{d}/theta-std.npy", theta_std)
            np.save(f"{d}/var-std.npy", variance_std)
            np.save(f"{d}/rank-std.npy", rank_std)
            np.save(f"{d}/rho.npy", rhos)
            np.save(f"{d}/tau.npy", taus)
            np.save(f"{d}/aucs.npy", aucs)
            np.save(f"{d}/accs.npy", accs)

            main_file.write(f"\n{key}, {base_acc}, {np.std(accs)}, {base_auc}, {np.std(aucs)}, {np.mean(rank_std)}")
            print(f"  [{key}] ACC: {base_acc:.4f} ±{np.std(accs):.4f}  AUC: {base_auc:.4f} ±{np.std(aucs):.4f}  <rank_std>: {np.mean(rank_std):.2f}")

    pre_annotations = load_dataset("MarcoLents/HaLO", "pre_study")

    print("\nPre-study bootstrap analysis")
    Path(f"{args.out_directory}/pre-study").mkdir(parents=True, exist_ok=True)
    with open(f"{args.out_directory}/pre-study/performance.csv", "w") as pre_file:
        pre_file.write("Annotator, ACC, sigma(ACC), AUC, sigma(AUC), <sigma(rank)>")
        for key, split in pre_annotations.items():
            Path(f"{args.out_directory}/pre-study/{key}").mkdir(parents=True, exist_ok=True)
            print(f"  Fitting [{key}] ({args.resamples} resamples)...", flush=True)

            (base_thetas, base_variances, base_ranks,
             theta_std, variance_std, rank_std,
             rhos, taus, base_auc, base_acc,
             aucs, accs) = bootstrap_analysis(pd.DataFrame(split), samples=args.resamples)

            d = f"{args.out_directory}/pre-study/{key}"
            np.save(f"{d}/theta.npy", base_thetas)
            np.save(f"{d}/var.npy", base_variances)
            np.save(f"{d}/ranks.npy", base_ranks)
            np.save(f"{d}/theta-std.npy", theta_std)
            np.save(f"{d}/var-std.npy", variance_std)
            np.save(f"{d}/rank-std.npy", rank_std)
            np.save(f"{d}/rho.npy", rhos)
            np.save(f"{d}/tau.npy", taus)
            np.save(f"{d}/aucs.npy", aucs)
            np.save(f"{d}/accs.npy", accs)

            pre_file.write(f"\n{key}, {base_acc}, {np.std(accs)}, {base_auc}, {np.std(aucs)}, {np.mean(rank_std)}")
            print(f"  [{key}] ACC: {base_acc:.4f} ±{np.std(accs):.4f}  AUC: {base_auc:.4f} ±{np.std(aucs):.4f}  <rank_std>: {np.mean(rank_std):.2f}")
            

    print("\nPre-study annotator agreements")
    with open(f"{args.out_directory}/pre-study/agreements.csv", "w") as agreement_file:
        agreement_file.write("Annotator1, Annotator2, rho, tau")
        for (key1, split1), (key2, split2) in combinations(pre_annotations.items(), 2):
            print(f"  Fitting {key1} vs {key2}...", flush=True)
            model1 = fit_model(split1)
            model2 = fit_model(split2)
    
            rho, tau = compare_models(model1, model2)

            agreement_file.write(f"\n{key1}, {key2}, {rho}, {tau}")
            print(f"  {key1} vs {key2}: rho={rho:.4f}  tau={tau:.4f}")

    if args.no_ml:
        print("Skipping analysis of machine-learning models")
        exit()

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

    # ML model analysis on the test split
    test_df = pd.DataFrame(main_annotations["test"])
    bt_model = fit_model(test_df)

    # Sorted IDs aligned with bt_model.thetas
    all_ids = sorted(set(test_df["sampleId1"]).union(set(test_df["sampleId2"])))
    id_to_path = {row["sampleId1"]: row["samplePath1"] for _, row in test_df.iterrows()}
    id_to_path.update({row["sampleId2"]: row["samplePath2"] for _, row in test_df.iterrows()})

    feat_dir = f"{args.data_dir}/features"
    out_dir = f"{args.out_directory}/ml-models"
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ml_models = {}
    for name in discover_models(args.models_dir):
        arch, dropout = parse_model_name(name)
        ml_models[name] = load_model(name, arch, dropout, args.models_dir).to(device)

    print("\nML model performance (test set)")
    with open(f"{out_dir}/performance.csv", "w") as ml_file:
        ml_file.write("model, rho, tau, AUC, ACC")
        for name, model in ml_models.items():
            ml_scores_dict = score_images(model, feat_dir, id_to_path, device)
            ml_scores = [ml_scores_dict[id_] for id_ in all_ids]

            rho = stats.spearmanr(bt_model.thetas, ml_scores).statistic
            tau = stats.kendalltau(bt_model.thetas, ml_scores).statistic

            auc, acc = eval_model_ml(model, main_annotations["test"], feat_dir, device)

            ml_file.write(f"\n{name}, {rho}, {tau}, {auc}, {acc}")
            print(f"  {name}: rho={rho:.4f}  tau={tau:.4f}  AUC={auc:.4f}  ACC={acc:.4f}")




if __name__ == "__main__":
    main()
