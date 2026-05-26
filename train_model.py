from datasets import load_dataset
from huggingface_hub import snapshot_download
from argparse import ArgumentParser
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from torch import optim
from halo_legibility.train import train
from halo_legibility.model import build_model
from halo_legibility.loader import HaLOPixtralFeatures

def main():
    parser = ArgumentParser(
        prog="Feature Experiments",
        description="Train the models described in 'Ranking handwriting images like a human'"
    )
    
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--model', default='m0')
    parser.add_argument('--name', default=None)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-e', '--epochs', type=int, default=2000)
    parser.add_argument('--optimizer', default="sgd")
    parser.add_argument('-d', '--dropout', type=float, default=0.25)
    parser.add_argument('-w', '--weight-decay', type=float, default=0)
    parser.add_argument('-l', '--learning-rate', type=float, default=1e-3)
    parser.add_argument('-p', '--patience', default=100, type=int)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
    print(f"Using {device} device (seed={args.seed})")
    
    # Download the features
    snapshot_download(
        repo_id="MarcoLents/HaLO",
        repo_type="dataset",
        allow_patterns="features/*/*.npy",
        local_dir=args.data_dir
    )

    if args.name is None:
        name = f"{args.model}_d{args.dropout}_w{args.weight_decay}_l{args.learning_rate}_s{args.seed}"
    else:
        name = args.name

    feat_dir = f"{args.data_dir}/features"


    model = build_model(args.model, args.dropout)

    model = model.to(device)

    g = torch.Generator()
    g.manual_seed(args.seed)

    dataset = load_dataset("MarcoLents/HaLO")
    train_loader = DataLoader(HaLOPixtralFeatures(dataset["train"], feat_dir), batch_size=args.batch_size, shuffle=True, generator=g)
    val_loader = DataLoader(HaLOPixtralFeatures(dataset["validation"], feat_dir), batch_size=args.batch_size)
    
    if args.optimizer.lower() == "adam":
        optimizer = optim.AdamW(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    elif args.optimizer.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), weight_decay=args.weight_decay, lr=args.learning_rate)
    else:
        ValueError(f"Unknown optimizer: {args.optimizer} possible values are 'Adam' and 'SGD'")

    train(name, model, args.epochs, train_loader, val_loader, device, optimizer=optimizer, patience=args.patience)
        
        
if __name__ == "__main__":
    main()
