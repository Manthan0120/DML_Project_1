import os
from train import train_mlp, train_cnn
from experiments import batch_size_experiment

def main():
    """Run the complete assignment."""
    print("Starting Assignment 1: Deep Learning with PyTorch")
    print("=" * 50)
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Train both models
    print("\n1. Training Models")
    print("-" * 20)
    mlp_model, mlp_df = train_mlp()
    cnn_model, cnn_df = train_cnn()
    
    # Run experiments
    print("\n2. Running Batch Size Experiment")
    print("-" * 35)
    exp_df = batch_size_experiment()
    
    # Final summary
    print("\n3. Final Results Summary")
    print("-" * 25)
    print(f"MLP final validation accuracy: {mlp_df['val_acc'].iloc[-1]:.4f}")
    print(f"CNN final validation accuracy: {cnn_df['val_acc'].iloc[-1]:.4f}")
    
    if cnn_df['val_acc'].iloc[-1] > mlp_df['val_acc'].iloc[-1]:
        print("CNN outperforms MLP as expected!")
    else:
        print("Warning: CNN did not outperform MLP")
    
    print("\nAll results saved to 'results/' directory")
    print("Assignment completed successfully!")

if __name__ == "__main__":
    main()
