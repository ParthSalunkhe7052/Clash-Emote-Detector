"""
PyTorch Neural Network Training for Pose Classification
Proper deep learning approach as per UPGRADE_PLAN.md
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime


class PoseDataset(Dataset):
    """PyTorch Dataset for pose landmarks with data augmentation"""
    def __init__(self, X, y, augment=False):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]
        
        if self.augment:
            x = self.augment_features(x)
        
        return x, y
    
    def augment_features(self, x):
        """
        Apply data augmentation to features
        - Add random noise
        - Scale features slightly
        - Flip left/right (horizontal flip)
        """
        x = x.clone()
        
        # Random noise (5% of feature range)
        if np.random.rand() > 0.5:
            noise = torch.randn_like(x) * 0.02
            x = x + noise
        
        # Random scaling (0.95 to 1.05)
        if np.random.rand() > 0.5:
            scale = 0.95 + np.random.rand() * 0.1
            x = x * scale
        
        # Horizontal flip (swap left/right features)
        # Features 0-1: shoulders, 2-3: wrist heights, 4-5: wrists, 6-7: elbows, 8-9: hips
        if np.random.rand() > 0.5:
            # Swap left-right pairs
            x_flipped = x.clone()
            # Swap shoulders (0 <-> 1)
            x_flipped[0], x_flipped[1] = -x[1], -x[0]
            # Swap wrist heights (2 <-> 3)
            x_flipped[2], x_flipped[3] = x[3], x[2]
            # Swap wrists (4 <-> 5)
            x_flipped[4], x_flipped[5] = -x[5], -x[4]
            # Swap elbows (6 <-> 7)
            x_flipped[6], x_flipped[7] = -x[7], -x[6]
            # Swap hips (8 <-> 9)
            x_flipped[8], x_flipped[9] = -x[9], -x[8]
            # Flip nose (10)
            x_flipped[10] = -x[10]
            # Swap arm lengths (11 <-> 12)
            x_flipped[11], x_flipped[12] = x[12], x[11]
            # Keep 13-17 same (widths, heights, symmetry)
            x = x_flipped
        
        return x


class PoseNeuralClassifier(nn.Module):
    """
    Neural Network for pose classification
    Architecture: Input -> Dense(256) -> Dropout -> Dense(128) -> Dropout -> Output
    """
    def __init__(self, input_dim=18, num_classes=7, dropout=0.3):
        super(PoseNeuralClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_model(X_train, y_train, X_val, y_val, num_classes=7, epochs=100, batch_size=32, lr=0.001):
    """
    Train the neural network
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        num_classes: Number of emote classes
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
    
    Returns:
        model: Trained model
        history: Training history
    """
    
    # Create datasets and dataloaders with augmentation
    train_dataset = PoseDataset(X_train, y_train, augment=True)  # Enable augmentation for training
    val_dataset = PoseDataset(X_val, y_val, augment=False)  # No augmentation for validation
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = PoseNeuralClassifier(input_dim=input_dim, num_classes=num_classes)
    
    # Calculate class weights for balanced training
    unique_classes, class_counts = np.unique(y_train, return_counts=True)
    class_weights = torch.FloatTensor([len(y_train) / (len(unique_classes) * count) for count in class_counts])
    
    # Loss and optimizer with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    
    print("="*60)
    print("TRAINING NEURAL NETWORK")
    print("="*60)
    print(f"Model Architecture:")
    print(f"  Input: {input_dim} features")
    print(f"  Hidden: 256 → 128 → 64")
    print(f"  Output: {num_classes} classes")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"\nTraining Config:")
    print(f"  Epochs: {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Learning Rate: {lr}")
    print(f"  Optimizer: AdamW")
    print("="*60)
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Metrics
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate metrics
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        val_loss /= len(val_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}]")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Best Val Acc: {best_val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print("\n" + "="*60)
    print(f"✅ TRAINING COMPLETE!")
    print(f"🏆 Best Validation Accuracy: {best_val_acc:.2f}%")
    print("="*60)
    
    return model, history


def evaluate_model(model, X_test, y_test, class_names):
    """Evaluate model on test set"""
    model.eval()
    
    X_test_tensor = torch.FloatTensor(X_test)
    
    with torch.no_grad():
        outputs = model(X_test_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    y_pred = predicted.numpy()
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    print(f"Accuracy: {accuracy*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return accuracy, y_pred, cm


def plot_training_history(history, save_path='results/training_curves.png'):
    """Plot training curves"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(history['val_loss'], label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy plot
    ax2.plot(history['train_acc'], label='Train Acc', color='blue')
    ax2.plot(history['val_acc'], label='Val Acc', color='red')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Training curves saved to {save_path}")
    plt.close()


def plot_confusion_matrix(cm, class_names, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Confusion matrix saved to {save_path}")
    plt.close()


def save_model_and_results(model, history, accuracy, save_dir='trained_model'):
    """Save model and training results"""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, 'pose_neural_classifier.pth')
    torch.save(model.state_dict(), model_path)
    print(f"\n💾 Model saved to {model_path}")
    
    # Save model architecture info
    arch_info = {
        'input_dim': list(model.parameters())[0].shape[1],
        'num_classes': list(model.parameters())[-1].shape[0],
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'architecture': str(model)
    }
    
    with open(os.path.join(save_dir, 'model_architecture.json'), 'w') as f:
        json.dump(arch_info, f, indent=2)
    
    # Save training results
    results = {
        'timestamp': datetime.now().isoformat(),
        'final_accuracy': float(accuracy),
        'best_val_acc': float(max(history['val_acc'])),
        'history': {
            'train_loss': [float(x) for x in history['train_loss']],
            'train_acc': [float(x) for x in history['train_acc']],
            'val_loss': [float(x) for x in history['val_loss']],
            'val_acc': [float(x) for x in history['val_acc']]
        }
    }
    
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to {save_dir}/training_results.json")


def main():
    """Main training script"""
    print("\n" + "="*60)
    print("PYTORCH NEURAL NETWORK TRAINING")
    print("Clash Emote Detector - Proper Deep Learning")
    print("="*60 + "\n")
    
    # Load data
    print("Loading training data...")
    X = np.load('pose_data/pose_features_latest.npy')
    y = np.load('pose_data/pose_labels_latest.npy')
    
    print(f"✅ Loaded {len(X)} samples")
    print(f"   Feature dimension: {X.shape[1]}")
    print(f"   Classes: {np.unique(y)}")
    
    # Class names
    class_names = [
        "Laughing", "Yawning", "Crying", "Taunting",
        "Arms Folded Laughing", "Hands Chest Kissing", "Hands Raised Screaming"
    ]
    
    # Show distribution
    print("\nData Distribution:")
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        print(f"  {class_names[label]}: {count} samples")
    
    # Split data
    print("\nSplitting data...")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.176, random_state=42, stratify=y_train_val
    )
    
    print(f"  Training: {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test: {len(X_test)} samples")
    
    # Train model
    print("\nStarting training...")
    model, history = train_model(
        X_train, y_train, X_val, y_val,
        num_classes=len(class_names),
        epochs=150,
        batch_size=32,
        lr=0.001
    )
    
    # Evaluate on test set
    accuracy, y_pred, cm = evaluate_model(model, X_test, y_test, class_names)
    
    # Plot results
    plot_training_history(history)
    plot_confusion_matrix(cm, class_names)
    
    # Save everything
    save_model_and_results(model, history, accuracy)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"✅ Model: trained_model/pose_neural_classifier.pth")
    print(f"✅ Accuracy: {accuracy*100:.2f}%")
    print(f"✅ Graphs: results/")
    print("="*60)
    print("\nNext step: Use this model in your webapp!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
