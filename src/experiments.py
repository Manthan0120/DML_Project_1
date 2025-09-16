import torch
import torch.nn as nn
import torch.optim as optim
import time
import pandas as pd
from models import MLP, SmallCIFARConvNet
from data import get_cifar10

def time_training(model, train_loader, valid_loader, epochs=5):
    """Time the training of a model for specified epochs."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    total_time = time.time() - start_time
    return total_time

def batch_size_experiment():
    """Run experiment to test training time vs batch size."""
    print("Running batch size experiment...")
    
    batch_sizes_to_test = [32, 64, 128, 256]
    num_epochs_for_exp = 5
    batch_results = []
    
    print("Testing MLP with different batch sizes...")
    
    for bs in batch_sizes_to_test:
        print(f"\nTesting batch size {bs}")
        
        train_data, valid_data, train_loader, valid_loader = get_cifar10(
            batch_size=bs,
            num_workers=2
        )
        
        mlp_model = MLP(num_classes=10, p=0.2)
        
        training_time = time_training(mlp_model, train_loader, valid_loader, epochs=num_epochs_for_exp)
        
        batch_results.append({
            'model': 'MLP',
            'batch_size': bs,
            'training_time': training_time,
            'time_per_epoch': training_time / num_epochs_for_exp
        })
        
        print(f"Training time: {training_time:.2f} seconds")

    print("\nTesting CNN with different batch sizes...")

    for bs in batch_sizes_to_test:
        print(f"\nTesting batch size {bs}")
        
        train_data, valid_data, train_loader, valid_loader = get_cifar10(
            batch_size=bs,
            num_workers=2
        )
        
        cnn_model = SmallCIFARConvNet(num_classes=10, p_drop=0.3)
        
        training_time = time_training(cnn_model, train_loader, valid_loader, epochs=num_epochs_for_exp)
        
        batch_results.append({
            'model': 'CNN', 
            'batch_size': bs,
            'training_time': training_time,
            'time_per_epoch': training_time / num_epochs_for_exp
        })
        
        print(f"Training time: {training_time:.2f} seconds")

    df_batch_exp = pd.DataFrame(batch_results)
    df_batch_exp.to_csv('results/batch_size_experiment.csv', index=False)
    
    print("\nBatch Size Experiment Results:")
    print(df_batch_exp)
    
    return df_batch_exp

if __name__ == "__main__":
    batch_size_experiment()
