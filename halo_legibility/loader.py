from torch.utils.data import Dataset
import os
import glob
import torch
import numpy as np


def score_images(model, feat_dir: str, id_to_path: dict, device: str) -> dict:
    """
    Run each unique image through model.classifier to get per-image scores.

    id_to_path: {sample_id: relative_path_to_npy}
    Returns: {sample_id: scalar_score}
    """
    model.eval()
    scores = {}
    with torch.no_grad():
        for img_id, path in id_to_path.items():
            feat = torch.tensor(np.load(f"{feat_dir}/{path}"), dtype=torch.float32).unsqueeze(0).to(device)
            score = model.classifier(feat).squeeze().item()
            scores[img_id] = score
    return scores


def index_files(pattern):
    index = {}
    for path in glob.glob(pattern, recursive=True):
        img_id = os.path.splitext(os.path.basename(path))[0]
        index[int(img_id)] = path
    return index


class HaLOPixtralFeatures(Dataset):
    def __init__(self, labels, feature_dir):
        self.labels = labels
        self.feature_dir = feature_dir
        
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feat1_path = self.labels[idx]["samplePath1"]
        feat1 = np.load(f"{self.feature_dir}/{feat1_path}")

        feat2_path = self.labels[idx]["samplePath2"]
        feat2 = np.load(f"{self.feature_dir}/{feat2_path}")
        
        label = 1.0 if self.labels[idx]["score"] == 1 else 0.0

        features = torch.tensor(np.array((feat1, feat2)))

        return features, label
