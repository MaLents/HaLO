import torch
import torch.nn as nn
import torch.nn.functional as F
from jaxtyping import Float
from einops import rearrange


class CustomLinear(nn.Module):
    def __init__(self, neuron_counts, dropout):
        super().__init__()
        self.fcs = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        last_neuron_count = neuron_counts[0]
        for neuron_count in neuron_counts[1:]:
            self.fcs.append(nn.Linear(last_neuron_count, neuron_count))
            last_neuron_count = neuron_count

    def forward(self, x: Float[torch.Tensor, "batch 2048"]):
        for fc in self.fcs[:-1]:
            x = self.dropout(F.relu(fc(x)))
        x = self.fcs[-1](x)
        return x


class ParallelFeatureNetwork(nn.Module):
    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier

    def handle_single(
        self,
        x: Float[torch.Tensor, "batch 2048"]
    ) -> Float[torch.Tensor, "batch"]:
        return self.classifier(x)

    def forward(
        self,
        x: Float[torch.Tensor, "batch 2 1024"],
    ) -> Float[torch.Tensor, "batch 2"]:
        x1, x2 = rearrange(x, "b i ... f -> i b ... f", i=2)
        x1 = self.handle_single(x1)
        x2 = self.handle_single(x2)
        out = torch.sigmoid(x1 - x2).squeeze()
        return out


def build_model(arch: str, dropout: float) -> ParallelFeatureNetwork:
    if arch == "m0":
        return ParallelFeatureNetwork(CustomLinear([1024, 1], dropout=dropout))
    elif arch == "m1":
        return ParallelFeatureNetwork(CustomLinear([1024, 256, 1], dropout=dropout))
    elif arch == "m2":
        return ParallelFeatureNetwork(CustomLinear([1024, 512, 256, 1], dropout=dropout))
    elif arch == "m3":
        return ParallelFeatureNetwork(CustomLinear([1024, 512, 256, 256, 1], dropout=dropout))
    else:
        raise ValueError(f"Unknown model: {arch}")


def load_model(name: str, arch: str, dropout: float, models_dir: str = "models") -> ParallelFeatureNetwork:
    model = build_model(arch, dropout)
    checkpoint = torch.load(f"{models_dir}/{name}/best.pt", map_location="cpu")
    model.load_state_dict(checkpoint)
    return model
