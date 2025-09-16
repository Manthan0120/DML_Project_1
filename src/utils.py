import torch
from types import SimpleNamespace

def accuracy(logits, targets):
    """Calculate accuracy from logits and targets."""
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()

def get_mlp_config():
    """Return configuration for MLP model."""
    return SimpleNamespace(
        dataset="cifar10", 
        batch_size=128,
        epochs=20,
        lr=1e-3,
        optimizer="adam",
        momentum=0.9,
        dropout=0.2,
        results_dir="results",
        run_name="mlp_notebook",
    )

def get_cnn_config():
    """Return configuration for CNN model."""
    return SimpleNamespace(
        dataset="cifar10",
        batch_size=128,
        epochs=20,
        lr=0.01,
        optimizer="sgd",
        momentum=0.9,
        weight_decay=5e-4,
        dropout=0.3,
        results_dir="results",
        run_name="cnn_sgd_mom09_lr001_wd5e4",
    )

def save_results_to_csv(records, output_path):
    """Save training records to CSV file."""
    import pandas as pd
    import os
    
    df = pd.DataFrame.from_records(records)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return df
