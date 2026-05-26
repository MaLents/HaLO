import logging
import pandas as pd
import numpy as np
import pymc as pm
from scipy.special import expit
import sklearn.metrics as metrics

DEFAULT_SAMPLE1_COL = "sampleId1"
DEFAULT_SAMPLE2_COL = "sampleId2"
DEFAULT_COMPARISON_COL = "score"

def create_index_map(ids):
    sorted_ids = sorted(ids)
    return {id_: idx for idx, id_ in enumerate(sorted_ids)}

def extract_comparisons(
    annotations: pd.DataFrame,
    sample1_col=DEFAULT_SAMPLE1_COL,
    sample2_col=DEFAULT_SAMPLE2_COL,
    comparison_col=DEFAULT_COMPARISON_COL,
    comparison_transform=lambda x: (x + 1) / 2,
):
    ids = list(set(annotations[sample1_col]).union(set(annotations[sample2_col])))
    index_map = create_index_map(ids)
    sample1_indices = np.array([index_map[id_] for id_ in annotations[sample1_col]])
    sample2_indices = np.array([index_map[id_] for id_ in annotations[sample2_col]])

    comparison_outcomes = comparison_transform(np.array(annotations[comparison_col]))

    return sample1_indices, sample2_indices, comparison_outcomes

def fit_model(annotations: pd.DataFrame, **kwargs):
    sample1_indices, sample2_indices, comparison_outcomes = extract_comparisons(annotations)
    model = BTModel()

    model.fit(sample1_indices, sample2_indices, comparison_outcomes, **kwargs)

    return model

def eval_model(model, annotations: pd.DataFrame):
    sample1_indices, sample2_indices, comparison_outcomes = extract_comparisons(annotations)

    predictions = model.predict(sample1_indices, sample2_indices)
    pred_labels = predictions > 0.5
    
    auc = metrics.roc_auc_score(comparison_outcomes, predictions)
    acc = metrics.accuracy_score(comparison_outcomes, pred_labels)

    return auc, acc
    

class BTModel:
    def __init__(self):
        self.trace = None
        self.thetas = None

    def fit(
        self,
        sample1_indices,
        sample2_indices,
        comparison_outcomes,
        random_seed=42,
        draws=2000,
        tune=1000,
        chains=4,
        data_backend="numpyro",
    ):
        with pm.Model() as model:
            num_indices = len(set(sample1_indices).union(set(sample2_indices)))
            rating = pm.Normal("rating", mu=0, sigma=1, shape=num_indices)

            p = pm.math.sigmoid(rating[sample1_indices] - rating[sample2_indices])
            
            outcome = pm.Bernoulli("outcome", p, observed=comparison_outcomes)

            logging.getLogger("pymc").setLevel(logging.ERROR)
            logging.getLogger("numba").setLevel(logging.ERROR)
            logging.getLogger("numpyro").setLevel(logging.ERROR)
            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                random_seed=random_seed,
                idata_backend=data_backend,
                progressbar=False,
            )

        self.trace = trace
        self.thetas = np.array(self.trace.posterior["rating"].mean(dim=("chain", "draw")).values)
        

    def rank(self):
        if self.trace is None:
            raise ValueError("You have to fit the model first")

        permutation = sorted(range(len(self.thetas)), key=lambda i: self.thetas[i])
        ranks = sorted(range(len(permutation)), key=lambda i: permutation[i])

        return np.array(ranks)

    def characterize_distributions(self):
        variances = self.trace.posterior["rating"].var(dim=("chain", "draw")).values
        ranks = self.rank()

        return self.thetas, variances, ranks

    def predict(self, sample1_indices, sample2_indices):
        differences = self.thetas[sample1_indices] - self.thetas[sample2_indices]
        return expit(differences)
