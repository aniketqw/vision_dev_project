"""
Test script for training_analytics package
Trains a simple CNN model on CIFAR-10 and logs all metrics using DebugLogger
"""

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Import DebugLogger from training_analytics package
from training_analytics.debug_logger import DebugLogger


# Define CNN Model
class SimpleCNN(pl.LightningModule):
    def __init__(self, learning_rate=0.001):
        super().__init__()
        self.learning_rate = learning_rate
        self.save_hyperparameters()
        
        # Convolutional layers
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = torch.nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = torch.nn.Linear(128 * 4 * 4, 256)
        self.fc2 = torch.nn.Linear(256, 10)
        
        # Dropout for regularization
        self.dropout = torch.nn.Dropout(0.5)
    
    def forward(self, x):
        # Conv block 1: 32 filters -> 16x16
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2: 64 filters -> 8x8
        x = self.pool(F.relu(self.conv2(x)))
        
        # Conv block 3: 128 filters -> 4x4
        x = self.pool(F.relu(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_loss', loss)
        self.log('accuracy', acc)
        return loss


def setup_data_loaders(batch_size=32):
    """Setup CIFAR-10 data loaders"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    val_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader


def main():
    """Main training function"""
    print("="*70)
    print("Testing Training Analytics Package")
    print("="*70)
    
    # Setup data
    print("\n📊 Loading CIFAR-10 dataset...")
    train_loader, val_loader = setup_data_loaders(batch_size=32)
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    
    # Initialize model
    print("\n🔧 Initializing SimpleCNN model...")
    model = SimpleCNN(learning_rate=0.001)
    print("✓ Model initialized")
    
    # Initialize DebugLogger
    print("\n📝 Initializing DebugLogger...")
    debug_logger = DebugLogger(save_dir='./logs')
    print("✓ DebugLogger initialized")
    
    # Create trainer with DebugLogger callback
    print("\n🚀 Creating Trainer...")
    trainer = pl.Trainer(
        max_epochs=3,
        callbacks=[debug_logger],
        enable_progress_bar=True,
        logger=False  # Disable default logger for cleaner output
    )
    print("✓ Trainer created")
    
    # Train model
    print("\n🎯 Starting training...\n")
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*70)
    print("✅ Training completed successfully!")
    print("="*70)
    print("\n📁 Check './logs/' folder for training_log_{timestamp}.json")
    print("\nYou can now:")
    print("  1. Analyze the JSON file with metrics")
    print("  2. View misclassified images")
    print("  3. Check gradient norms and learning rates")
    print("  4. Review batch-wise metrics")
    print("="*70)


if __name__ == "__main__":
    main()
