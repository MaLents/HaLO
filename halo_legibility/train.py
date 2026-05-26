import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score
from itertools import islice
import numpy as np
from tqdm import tqdm
from pathlib import Path

def train(name, model, epochs, train_loader, val_loader, device, optimizer, criterion=nn.CrossEntropyLoss(), patience=None):
    
    writer = SummaryWriter(f"runs/{name}")

    Path(f"models/{name}").mkdir(parents=True, exist_ok=True)

    best_val_loss = float('inf')
    best_epoch = -1
    epochs_without_improvement = 0
    
    batch_count = 0
    for epoch in tqdm(range(epochs), desc="Epochs"):
    
        t = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)
        for i, (train_features, train_labels) in enumerate(t):
    
            optimizer.zero_grad()
    
            outputs = model(train_features.to(device))
    
            loss = criterion(outputs, train_labels.to(device))
            loss.backward()
    
            optimizer.step()
            t.set_postfix(Loss=loss.item())
            writer.add_scalar("Batch/Loss/train", loss.item(), batch_count)
    
            if batch_count % 50 == 0:
                val_loss, val_auc, val_acc = evaluate_model(model, val_loader, criterion, device, num_batches=10)
                writer.add_scalar("Batch/Loss/val", val_loss, batch_count)
                writer.add_scalar("Batch/AUC/val", val_auc, batch_count)
                writer.add_scalar("Batch/ACC/val", val_acc, batch_count)
            
            batch_count += 1
        val_loss, val_auc, val_acc = evaluate_model(model, val_loader, criterion, device)
        writer.add_scalar("Epoch/Loss/val", val_loss, epoch)
        writer.add_scalar("Epoch/AUC/val", val_auc, epoch)
        writer.add_scalar("Epoch/ACC/val", val_acc, epoch)
        
        train_loss, train_auc, train_acc = evaluate_model(model, train_loader, criterion, device)
        writer.add_scalar("Epoch/Loss/train", train_loss, epoch)
        writer.add_scalar("Epoch/AUC/train", train_auc, epoch)
        writer.add_scalar("Epoch/ACC/train", train_acc, epoch)
    
        torch.save(model.state_dict(), f"models/{name}/epoch_{epoch}.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(model.state_dict(), f"models/{name}/best.pt")
        else:
            epochs_without_improvement += 1

        if patience is not None and epochs_without_improvement >= patience:
            tqdm.write(f"Early stopping at epoch {epoch+1} (best val loss: {best_val_loss:.4f} at epoch {best_epoch})")
            break



def evaluate_model(model, loader, criterion, device, num_batches=None):
    if not num_batches:
        num_batches = len(loader)
        
    was_training = model.training
    model.eval()
    
    running_loss = 0.0

    true_labels = []
    predictions = []
    
    iterator = islice(loader, num_batches)
    t = tqdm(enumerate(iterator), desc="Validating", leave=False, total=num_batches)
    for i, data in t:
        images, batch_labels = data

        true_labels.extend(batch_labels.cpu())
        

        with torch.no_grad():
            outputs = model(images.to(device))

        predictions.extend(outputs.cpu())
    
        loss = criterion(outputs, batch_labels.to(device))
    
        running_loss += loss.item()
        t.set_postfix(Loss=loss.item())

    true_labels = np.array(true_labels).astype(int)
    predictions = np.array(predictions)

    auc = roc_auc_score(true_labels, predictions)
    acc = accuracy_score(true_labels, np.rint(predictions))
    
    if was_training:
        model.train()
    return running_loss / num_batches, auc, acc
