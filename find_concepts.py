from datasets import load_dataset
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, mean_squared_error
from argparse import ArgumentParser
import numpy as np
import os
import csv


CONCEPTS = ["isWrittenInPureCursive", "isStrokeThin", "containsTypo", "containsCorrection"]


def load_features(split, feat_dir, concept):
    X = np.array([np.load(f"{feat_dir}/{row['samplePath']}") for row in split])
    y = np.array([float(row[concept]) for row in split])
    return X, y


def train_probe(X, y):
    clf = LogisticRegression(solver='liblinear', class_weight='balanced', max_iter=2000)
    grid = GridSearchCV(clf, {'C': [0.01, 0.1, 1, 10, 100]}, cv=5, scoring='roc_auc')
    grid.fit(X, y)
    print(f"  Best C: {grid.best_params_['C']}")
    return grid.best_estimator_


def train_ridge(X, y):
    grid = GridSearchCV(
        Ridge(fit_intercept=True, solver='svd'),
        {'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 500]},
        cv=5,
        scoring='neg_mean_squared_error'
    )
    grid.fit(X, y)
    cv_mse = -grid.best_score_
    print(f"  Best alpha: {grid.best_params_['alpha']}  CV MSE: {cv_mse:.4f}  CV RMSE: {np.sqrt(cv_mse):.4f}")
    return grid.best_estimator_, cv_mse


def evaluate(clf, X, y, split_name):
    y_pred = clf.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, y_pred)
    ap = average_precision_score(y, y_pred)
    acc = accuracy_score(y, clf.predict(X))
    print(f"  [{split_name}] AUC: {auc:.4f}  AP: {ap:.4f}  ACC: {acc:.4f}")
    return auc, ap, acc


def main():
    parser = ArgumentParser(
        prog="find_concepts",
        description="Train logistic regression probes for binary handwriting characteristics"
    )
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('-o', '--out-directory', default='results')
    args = parser.parse_args()

    snapshot_download(
        repo_id="MarcoLents/HaLO",
        repo_type="dataset",
        allow_patterns="features/*/*.npy",
        local_dir=args.data_dir
    )

    feat_dir = f"{args.data_dir}/features"
    concepts_dir = os.path.join(args.out_directory, "concepts")
    os.makedirs(concepts_dir, exist_ok=True)

    # --- Binary concept probes ---
    dataset = load_dataset("MarcoLents/HaLO", "characteristics")

    results = []
    for concept in CONCEPTS:
        print(f"\nTraining probe for: {concept}")
        X_concept_train, y_concept_train = load_features(dataset["train"], feat_dir, concept)
        X_concept_val, y_concept_val = load_features(dataset["validation"], feat_dir, concept)
        X_concept_test, y_concept_test = load_features(dataset["test"], feat_dir, concept)

        clf = train_probe(X_concept_train, y_concept_train)

        val_auc, val_ap, val_acc = evaluate(clf, X_concept_val, y_concept_val, "val")
        test_auc, test_ap, test_acc = evaluate(clf, X_concept_test, y_concept_test, "test")

        np.save(os.path.join(concepts_dir, f"{concept}.npy"), clf.coef_[0])
        results.append({
            "concept": concept,
            "val_AUC": val_auc, "val_AP": val_ap, "val_ACC": val_acc,
            "test_AUC": test_auc, "test_AP": test_ap, "test_ACC": test_acc,
        })

    perf_path = os.path.join(concepts_dir, "performance.csv")
    with open(perf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept", "val_AUC", "val_AP", "val_ACC", "test_AUC", "test_AP", "test_ACC"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved performance to {perf_path}")

    # --- Aspect ratio regression ---
    print("\nTraining Ridge regression for: aspectRatio")
    ar_dataset = load_dataset("MarcoLents/HaLO", "aspect_ratio")

    X_train = np.array([np.load(f"{feat_dir}/{row['samplePath']}") for row in ar_dataset["train"]])
    y_train = np.array([float(row["aspectRatio"]) for row in ar_dataset["train"]])
    X_val = np.array([np.load(f"{feat_dir}/{row['samplePath']}") for row in ar_dataset["validation"]])
    y_val = np.array([float(row["aspectRatio"]) for row in ar_dataset["validation"]])
    X_test = np.array([np.load(f"{feat_dir}/{row['samplePath']}") for row in ar_dataset["test"]])
    y_test = np.array([float(row["aspectRatio"]) for row in ar_dataset["test"]])

    ridge, cv_mse = train_ridge(X_train, y_train)

    val_mse, test_mse = [
        mean_squared_error(y, ridge.predict(X))
        for X, y in [(X_val, y_val), (X_test, y_test)]
    ]
    for mse, split in [(val_mse, "val"), (test_mse, "test")]:
        print(f"  [{split}] MSE: {mse:.4f}  RMSE: {np.sqrt(mse):.4f}")

    np.save(os.path.join(concepts_dir, "aspectRatio.npy"), ridge.coef_)

    ar_perf_path = os.path.join(concepts_dir, "performance_aspect_ratio.csv")
    with open(ar_perf_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept", "val_MSE", "val_RMSE", "test_MSE", "test_RMSE"])
        writer.writeheader()
        writer.writerow({
            "concept": "aspectRatio",
            "val_MSE": val_mse, "val_RMSE": np.sqrt(val_mse),
            "test_MSE": test_mse, "test_RMSE": np.sqrt(test_mse),
        })

    print(f"Saved aspect ratio performance to {ar_perf_path}")


if __name__ == "__main__":
    main()
