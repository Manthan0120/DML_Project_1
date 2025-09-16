import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
import os
from models import MLP, SmallCIFARConvNet
from data import get_cifar10
from utils import accuracy, get_mlp_config, get_cnn_config, save_results_to_csv

def run_epoch(model, loader, criterion, optimizer, train=True):
    """Run one epoch of training or validation."""
    if train:
        model.train()
    else:
        model.eval()
    
    total_loss, total_acc, total_n = 0.0, 0.0, 0
    
    for X, y in loader:
        if train:
            optimizer.zero_grad(set_to_none=True)
        
        with torch.set_grad_enabled(train):
            logits = model(X)
            loss = criterion(logits, y)
            
            if train:
                loss.backward()
                optimizer.step()
        
        bs = y.size(0)
        total_loss += loss.detach().item() * bs
        total_acc += accuracy(logits.detach(), y) * bs
        total_n += bs
    
    return total_loss / total_n, total_acc / total_n

def train_mlp():
    """Train MLP model."""
    print("Training MLP model...")
    cfg = get_mlp_config()
    
    train_data, valid_data, train_loader, valid_loader = get_cifar10(
        root=".",
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=True,
        use_normalize=True,
        use_augment=True, 
    )
    
    num_classes = 10
    model = MLP(num_classes=num_classes, p=cfg.dropout)
    
    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer == "adam":
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    elif cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum)
    else:
        raise ValueError(f"Unknown optimizer: {cfg.optimizer}")
    
    records = []
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, optimizer, train=False)
        elapsed = time.time() - start_time

        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": elapsed,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "optimizer": cfg.optimizer,
            "dataset": cfg.dataset,
            "model": "MLP"
        }
        records.append(rec)
        print(json.dumps(rec))

    output_path = os.path.join(cfg.results_dir, f"{cfg.run_name}_metrics.csv")
    df = save_results_to_csv(records, output_path)
    
    return model, df

def train_cnn():
    """Train CNN model."""
    print("Training CNN model...")
    cfg = get_cnn_config()
    
    train_data, valid_data, train_loader, valid_loader = get_cifar10(
        root=".",
        batch_size=cfg.batch_size,
        num_workers=2,
        pin_memory=True,
        use_normalize=True,
        use_augment=True,
    )
    
    model = SmallCIFARConvNet(num_classes=10, p_drop=cfg.dropout)
    
    criterion = nn.CrossEntropyLoss()
    if cfg.optimizer == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=cfg.lr, 
                             momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    records = []
    start = time.time()
    
    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, train=True)
        val_loss, val_acc = run_epoch(model, valid_loader, criterion, optimizer, train=False)
        
        if scheduler is not None:
            scheduler.step()
            
        rec = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "elapsed_sec": time.time() - start,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "optimizer": cfg.optimizer,
            "dataset": cfg.dataset,
            "model": "CNN",
            "weight_decay": cfg.weight_decay,
        }
        records.append(rec)
        print(json.dumps(rec))
    
    output_path = os.path.join(cfg.results_dir, f"{cfg.run_name}_metrics.csv")
    df = save_results_to_csv(records, output_path)
    
    return model, df

if __name__ == "__main__":
    mlp_model, mlp_df = train_mlp()
    cnn_model, cnn_df = train_cnn()
    
    print("\nTraining completed!")
    print(f"MLP final validation accuracy: {mlp_df['val_acc'].iloc[-1]:.4f}")
    print(f"CNN final validation accuracy: {cnn_df['val_acc'].iloc[-1]:.4f}")
