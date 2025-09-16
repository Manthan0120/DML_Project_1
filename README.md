# DML Assignment 1: Deep Learning with PyTorch

## Project Overview
This project implements and compares two deep learning architectures for CIFAR-10 image classification:
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Network (CNN)

Additionally, it includes a batch size optimization experiment to analyze computational efficiency trade-offs.

## System Requirements
- **Python Version**: 3.12.9 (run `python --version` to check)
- **PyTorch Version**: 2.3.1+cpu
- **Operating System**: Compatible with Windows, macOS, and Linux

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install required dependencies:
```bash
cd src
pip install -r requirements.txt
```

## Project Structure
```
├── src/                        # Source code directory
│   ├── data.py                 # Data loading and preprocessing
│   ├── experiments.py          # Batch size optimization experiments
│   ├── main.py                 # Main execution script
│   ├── models.py               # MLP and CNN model definitions
│   ├── train.py                # Training and validation logic
│   ├── utils.py                # Utility functions
│   └── requirements.txt        # Python dependencies
├── obtained_results/            # Output directory for results
│   ├── *.csv                   # Training metrics and experiment results
│   └── *.png/.pdf              # Generated plots (if any)
├── dml_project_report.pdf      # Final academic report
└── README.md                   # This file
```

## How to Run

### Complete Training Pipeline
Run the full experiment (both models + batch size analysis):
```bash
cd src
python main.py
```

### Individual Components
You can also run specific components separately:

**Train MLP only:**
```bash
python main.py --model mlp
```

**Train CNN only:**
```bash
python main.py --model cnn
```

**Run batch size experiment:**
```bash
python experiments.py
```

### Key Configuration Parameters

The models use the following configurations:

**MLP Configuration:**
- Architecture: 3072 96 1024 96 512 96 10
- Optimizer: Adam (lr=1e-3)
- Dropout: 0.2
- Epochs: 20

**CNN Configuration:**
- Architecture: 3 conv blocks + classifier
- Optimizer: SGD (lr=0.01, momentum=0.9, weight_decay=5e-4)
- Dropout: 0.3
- Epochs: 20

## Code Structure

### Core Files Description

- **`main.py`**: Entry point script that orchestrates the entire training pipeline
- **`models.py`**: Contains MLP and CNN class definitions with proper PyTorch inheritance
- **`data.py`**: Handles CIFAR-10 dataset loading, preprocessing, and data augmentation
- **`train.py`**: Implements training/validation loops and metric computation
- **`experiments.py`**: Batch size optimization experiments across different configurations
- **`utils.py`**: Helper functions for logging, plotting, and result saving

## Expected Results

### Model Performance:
- **MLP**: ~43.8% validation accuracy
- **CNN**: ~80.8% validation accuracy (significantly outperforms MLP as required)

### Training Time:
- **MLP**: ~12 minutes (20 epochs)
- **CNN**: ~53 minutes (20 epochs)

### Batch Size Experiment:
Tests batch sizes [32, 64, 128, 256] on both architectures for 5 epochs each.

## Output Files

Results are saved in the `obtained_result/` directory:
- `mlp_metrics.csv`: MLP training metrics (loss, accuracy per epoch)
- `cnn_metrics.csv`: CNN training metrics (loss, accuracy per epoch)
- `batch_experiment.csv`: Batch size comparison results
- Learning curve plots (if generated)

## Dataset Information
- **Dataset**: CIFAR-10 (automatically downloaded on first run)
- **Training Images**: 50,000
- **Validation Images**: 10,000
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Image Size**: 32323

## Dependencies

The `requirements.txt` includes:
```
torch>=2.3.1
torchvision
pandas
numpy
matplotlib
seaborn
```

To generate your own requirements file:
```bash
python -m pip freeze > requirements.txt
```

## Computational Requirements
- **CPU**: The code runs on CPU (no GPU required)
- **Memory**: Approximately 4-8GB RAM recommended
- **Storage**: ~500MB for CIFAR-10 dataset + results

## Troubleshooting

**Common Issues:**

1. **Import Errors**: Ensure you're running from the `src/` directory
2. **Memory Error**: Reduce batch size in model configurations
3. **CIFAR-10 Download Issues**: Check internet connection; dataset downloads automatically
4. **Results Directory**: Code automatically creates `obtained_result/` directory
5. **Module Not Found**: Verify all dependencies are installed via `pip install -r requirements.txt`

**Performance Notes:**
- CNN training significantly longer than MLP due to computational complexity
- Batch size 256 may cause memory issues on resource-constrained systems
- All experiments designed for CPU execution

## Assignment Compliance

This implementation satisfies all assignment requirements:
- 7 MLP with 2+ hidden layers using standard PyTorch
- 7 CNN with 2+ hidden layers using standard PyTorch  
- 7 CNN outperforms MLP (80.8% vs 43.8% accuracy)
- 7 CIFAR-10 dataset with proper train/validation split
- 7 CrossEntropyLoss for classification
- 7 Additional batch size optimization experiment
- 7 Learning curves showing train/validation metrics
- 7 Modular code structure with proper separation of concerns
- 7 Comprehensive data logging and CSV output
- 7 Academic report following IEEE format

## Usage Examples

**Basic usage:**
```bash
cd src
python main.py
```

**Custom configuration:**
```bash
python main.py --epochs 30 --batch-size 64
```

**Run specific experiment:**
```bash
python experiments.py --models mlp cnn --batch-sizes 32 64 128
```

## Results Interpretation

1. **Learning Curves**: Check `obtained_result/` for training progress
2. **Final Accuracies**: CNN should significantly outperform MLP
3. **Batch Size Analysis**: Optimal batch size varies by architecture
4. **Generalization**: CNN shows better validation performance stability

## Contact Information
For questions or issues, please refer to the course Canvas discussion page.

---
**Course**: CS 595-003 ("Decentralized ML Systems")  
**Assignment**: Deep Learning Assignment #1  
**Instructor**: Dr. Nathaniel Hudson  
**Due Date**: September 15th, 2025 at 11:59pm CDT  
**Submission**: Submit .zip file containing all code, results, and report
