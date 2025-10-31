"""
Enhanced Visual Feature Training Script
Trains on 54-D features (pose + hands + face) for better emotion detection
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
import sys
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class EnhancedDataset(Dataset):
    """PyTorch Dataset for enhanced visual features with data augmentation"""
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
        Apply data augmentation to 54-D features
        - Add random noise
        - Scale features slightly
        - Horizontal flip (swap left/right)
        """
        x = x.clone()
        
        # Random noise (3% of feature range)
        if np.random.rand() > 0.5:
            noise = torch.randn_like(x) * 0.015
            x = x + noise
        
        # Random scaling (0.97 to 1.03)
        if np.random.rand() > 0.5:
            scale = 0.97 + np.random.rand() * 0.06
            x = x * scale
        
        # Horizontal flip (swap left/right features)
        if np.random.rand() > 0.5:
            x_flipped = x.clone()
            
            # Pose features (0-17): swap left-right pairs
            # Shoulders (0 <-> 1)
            x_flipped[0], x_flipped[1] = -x[1], -x[0]
            # Wrist heights (2 <-> 3)
            x_flipped[2], x_flipped[3] = x[3], x[2]
            # Wrists (4 <-> 5)
            x_flipped[4], x_flipped[5] = -x[5], -x[4]
            # Elbows (6 <-> 7)
            x_flipped[6], x_flipped[7] = -x[7], -x[6]
            # Upper arms (9 <-> 10)
            x_flipped[9], x_flipped[10] = x[10], x[9]
            # Forearms (11 <-> 12)
            x_flipped[11], x_flipped[12] = x[12], x[11]
            # Hips (13 <-> 14)
            x_flipped[13], x_flipped[14] = -x[14], -x[13]
            # Nose (16)
            x_flipped[16] = -x[16]
            
            # Hand features (18-41): swap left hand (18-29) with right hand (30-41)
            x_flipped[18:30] = x[30:42].clone()
            x_flipped[30:42] = x[18:30].clone()
            # Flip horizontal positions
            for i in [18, 20, 30, 32]:  # X positions
                x_flipped[i] = -x_flipped[i]
            
            # Face features (42-53): swap left-right
            # Mouth corners (2 <-> 3)
            x_flipped[44], x_flipped[45] = x[45], x[44]
            # Eyes (4 <-> 5)
            x_flipped[46], x_flipped[47] = x[47], x[46]
            # Eyebrows (6 <-> 7)
            x_flipped[48], x_flipped[49] = x[49], x[48]
            
            x = x_flipped
        
        return x


class EnhancedNeuralClassifier(nn.Module):
    """
    Enhanced Neural Network for visual emotion classification
    Input: 54-D (18 pose + 24 hands + 12 face)
    Architecture: Deeper network to handle richer features
    """
    def __init__(self, input_dim=54, num_classes=7, dropout=0.3):
        super(EnhancedNeuralClassifier, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: 54 -> 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 2: 256 -> 192
            nn.Linear(256, 192),
            nn.BatchNorm1d(192),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3: 192 -> 128
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 4: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7),
            
            # Output layer
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    return total_loss / len(loader), correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels


def main():
    print("="*60)
    print("  ENHANCED VISUAL EMOTION CLASSIFIER TRAINING")
    print("  54-D Features: Pose (18) + Hands (24) + Face (12)")
    print("="*60)
    
    # Load data
    data_path = 'pose_data/pose_features_latest.npy'
    labels_path = 'pose_data/pose_labels_latest.npy'
    
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        print("❌ Training data not found!")
        print("   Run: python collect_training_data.py")
        print("   Then: python training/export_collected_data.py")
        return
    
    X = np.load(data_path)
    y = np.load(labels_path)
    
    print(f"\n📊 Dataset loaded:")
    print(f"   Samples: {len(X)}")
    print(f"   Feature dimension: {X.shape[1]}")
    print(f"   Classes: {len(np.unique(y))}")
    
    # Check feature dimension
    if X.shape[1] != 54:
        print(f"\n⚠️  WARNING: Expected 54-D features, got {X.shape[1]}-D")
        print("   Make sure you collected data with the enhanced feature extractor!")
        if X.shape[1] == 18:
            print("   Your data appears to be old 18-D pose-only features.")
            print("   Please re-collect data using the updated collect_training_data.py")
            return
    
    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📈 Class distribution:")
    class_names = ["Laughing", "Yawning", "Crying", "Taunting", 
                   "Arms Folded Laughing", "Hands Chest Kissing", "Hands Raised Screaming"]
    for label, count in zip(unique, counts):
        print(f"   {class_names[label]}: {count} samples")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"\n📂 Data splits:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val:   {len(X_val)} samples")
    print(f"   Test:  {len(X_test)} samples")
    
    # Create datasets
    train_dataset = EnhancedDataset(X_train, y_train, augment=True)
    val_dataset = EnhancedDataset(X_val, y_val, augment=False)
    test_dataset = EnhancedDataset(X_test, y_test, augment=False)
    
    # Create dataloaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Model setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    model = EnhancedNeuralClassifier(input_dim=54, num_classes=7, dropout=0.3).to(device)
    
    # Class weights for imbalanced data
    class_weights = torch.FloatTensor([len(y) / (len(unique) * count) for count in counts]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    
    # Training
    num_epochs = 150
    best_val_acc = 0
    patience = 25
    patience_counter = 0
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("="*60)
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        scheduler.step(val_loss)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.2f}% | "
                  f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'enhanced_visual_classifier.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break
    
    print("="*60)
    print(f"✅ Training complete! Best val accuracy: {best_val_acc*100:.2f}%")
    
    # Load best model and evaluate on test set
    model.load_state_dict(torch.load('enhanced_visual_classifier.pth'))
    test_loss, test_acc, test_preds, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\n📊 Test Set Performance:")
    print(f"   Accuracy: {test_acc*100:.2f}%")
    print(f"   Loss: {test_loss:.4f}")
    
    # Classification report
    print(f"\n📋 Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=class_names))
    
    # Confusion matrix
    cm = confusion_matrix(test_labels, test_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Enhanced Visual Classifier')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('results/enhanced_confusion_matrix.png', dpi=150)
    print("   Saved confusion matrix to results/enhanced_confusion_matrix.png")
    
    # Training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot([acc*100 for acc in train_accs], label='Train Acc')
    plt.plot([acc*100 for acc in val_accs], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/enhanced_training_curves.png', dpi=150)
    print("   Saved training curves to results/enhanced_training_curves.png")
    
    # Save model info
    model_info = {
        'model_type': 'Enhanced Visual Classifier',
        'input_dim': 54,
        'num_classes': 7,
        'architecture': '54->256->192->128->64->7',
        'best_val_accuracy': float(best_val_acc),
        'test_accuracy': float(test_acc),
        'training_samples': len(X_train),
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_breakdown': {
            'pose': 18,
            'hands': 24,
            'face': 12
        }
    }
    
    with open('enhanced_model_info.json', 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n✅ Model saved as: enhanced_visual_classifier.pth")
    print(f"✅ Model info saved as: enhanced_model_info.json")
    print("\n🎯 Next steps:")
    print("   1. Test the model in the web app")
    print("   2. Collect more data if accuracy is low for specific emotes")
    print("   3. Consider fine-tuning hyperparameters if needed")


if __name__ == "__main__":
    # Create results directory
    os.makedirs('results', exist_ok=True)
    main()
