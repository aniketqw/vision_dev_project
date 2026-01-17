# Training Analytics Package

Comprehensive logging and debugging callback for PyTorch Lightning training.

## Features

✅ **Epoch-level Metrics**
- Training and validation loss
- Accuracy, Precision, Recall, F1-Score
- Gradient norms
- Learning rates
- Epoch timing

✅ **Batch-level Metrics**
- Per-batch accuracy and loss
- Identifies problematic batches

✅ **Class-wise Metrics**
- Per-class precision, recall, F1-score
- Support per class

✅ **Input Data Statistics**
- Mean, Std, Min, Max per batch
- Ensures proper data normalization

✅ **Misclassified Samples**
- Base64-encoded images
- True and predicted labels
- Epoch information

✅ **Gradient Monitoring**
- L2 norm of gradients per epoch
- Detects vanishing/exploding gradients

## Usage

```python
from exp.helperFunc.training_analytics import DebugLogger
import pytorch_lightning as pl

# Create logger instance
debug_logger = DebugLogger(save_dir='./logs')

# Add to trainer
trainer = pl.Trainer(
    max_epochs=10,
    callbacks=[debug_logger]
)

# Train model
trainer.fit(model, train_loader, val_loader)

# All metrics saved to: ./logs/training_log_{timestamp}.json
```

## Output JSON Structure

```json
{
  "summary": {
    "timestamp": "20260117_230000",
    "total_epochs": 10,
    "best_accuracy": 0.85,
    "best_epoch": 7,
    "final_train_loss": 0.234,
    "final_val_loss": 0.456,
    "total_misclassified": 1500,
    "total_time_seconds": 3600.5,
    "avg_gradient_norm": 0.0234
  },
  "epochs": [...],
  "batch_metrics": [...],
  "class_metrics": [...],
  "gradient_norms": [...],
  "learning_rates": [...],
  "epoch_times": [...],
  "input_statistics": [...],
  "misclassified_samples": [...]
}
```

## Requirements

- pytorch-lightning
- torch
- scikit-learn
- numpy
- Pillow

## File Structure

```
training_analytics/
├── __init__.py
├── debug_logger.py
└── README.md
```

## Author

Vision Dev Project
