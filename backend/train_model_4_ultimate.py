"""
Ultimate Model 4 Training Script
Uses 128-D MediaPipe embeddings with advanced architecture
Combines all available data for maximum performance
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Import our embedder
from .mediapipe_embedder import MediaPipePoseEmbedder

print("=" * 60)
print("🚀 ULTIMATE MODEL 4 TRAINING")
print("=" * 60)

# Configuration
DATA_DIR = Path('pose_data_v2')
MODEL_NAME = 'pose_model_4_ultimate.pth'
INFO_NAME = 'pose_model_4_ultimate_info.json'
BATCH_SIZE = 32
EPOCHS = 200  # More epochs for better training
LEARNING_RATE = 0.001
PATIENCE = 25  # Early stopping patience

# Emote mapping
EMOTE_NAMES = {
    0: "crying",
    1: "laughing", 
    2: "taunting",
    3: "yawning",
    4: "arms_folded_laughing",
    5: "hands_chest_kissing",
    6: "hands_raised_screaming"
}

class UltimateModel(nn.Module):
    """
    Ultimate neural network architecture for Model 4
    - Deeper network
    - Residual connections
    - Attention mechanism
    - Advanced regularization
    """
    def __init__(self, input_dim=128, num_classes=7):
        super(UltimateModel, self).__init__()
        
        # Input layer with batch norm
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # First block (128 -> 512)
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.4)
        
        # Second block (512 -> 512) with residual
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.4)
        
        # Third block (512 -> 256)
        self.fc3 = nn.Linear(512, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.3)
        
        # Fourth block (256 -> 256) with residual
        self.fc4 = nn.Linear(256, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(0.3)
        
        # Fifth block (256 -> 128)
        self.fc5 = nn.Linear(256, 128)
        self.bn5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(0.2)
        
        # Attention layer
        self.attention = nn.Linear(128, 128)
        self.attention_softmax = nn.Softmax(dim=1)
        
        # Output layer
        self.fc_out = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Input normalization
        x = self.input_bn(x)
        
        # Block 1
        x1 = self.fc1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.dropout1(x1)
        
        # Block 2 with residual
        x2 = self.fc2(x1)
        x2 = self.bn2(x2)
        x2 = self.relu2(x2)
        x2 = self.dropout2(x2)
        x2 = x2 + x1  # Residual connection
        
        # Block 3
        x3 = self.fc3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x3 = self.dropout3(x3)
        
        # Block 4 with residual
        x4 = self.fc4(x3)
        x4 = self.bn4(x4)
        x4 = self.relu4(x4)
        x4 = self.dropout4(x4)
        x4 = x4 + x3  # Residual connection
        
        # Block 5
        x5 = self.fc5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)
        x5 = self.dropout5(x5)
        
        # Attention mechanism
        attention_weights = self.attention(x5)
        attention_weights = self.attention_softmax(attention_weights)
        x5 = x5 * attention_weights
        
        # Output
        out = self.fc_out(x5)
        return out

class PoseDataset(Dataset):
    """PyTorch dataset for pose embeddings"""
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

def load_data():
    """Load all .npz files and extract 128-D embeddings"""
    print("\n📂 Loading data from pose_data_v2/...")
    
    embedder = MediaPipePoseEmbedder()
    all_embeddings = []
    all_labels = []
    
    # Count samples per emote
    samples_per_emote = {i: 0 for i in range(7)}
    
    for emote_id, emote_name in EMOTE_NAMES.items():
        emote_files = sorted(DATA_DIR.glob(f"{emote_name}_*.npz"))
        print(f"  {emote_name}: {len(emote_files)} samples")
        
        for i, npz_file in enumerate(emote_files):
            if i % 50 == 0:
                print(f"    Processing {i}/{len(emote_files)}...")
            try:
                data = np.load(npz_file)
                pose = data['pose']  # Shape: (33, 3)
                visibility = data['visibility']  # Shape: (33,)
                
                # Combine pose and visibility to get (33, 4)
                landmarks = np.column_stack([pose, visibility])
                
                # Extract 128-D embedding
                embedding = embedder.extract_features(landmarks)
                
                all_embeddings.append(embedding)
                all_labels.append(emote_id)
                samples_per_emote[emote_id] += 1
                
            except Exception as e:
                print(f"    ⚠️ Error loading {npz_file.name}: {e}")
                continue
    
    print(f"\n✅ Loaded {len(all_embeddings)} total samples")
    print(f"📊 Samples per emote: {samples_per_emote}")
    
    return np.array(all_embeddings), np.array(all_labels), samples_per_emote

def create_weighted_sampler(labels):
    """Create weighted sampler for balanced training"""
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights))

def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for embeddings, labels in train_loader:
        embeddings, labels = embeddings.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(train_loader), 100. * correct / total

def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for embeddings, labels in val_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            
            outputs = model(embeddings)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(val_loader), 100. * correct / total

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Device: {device}")
    
    # Load data
    embeddings, labels, samples_per_emote = load_data()
    
    # Check for class imbalance
    print(f"\n⚖️  Checking class balance...")
    for emote_id, count in samples_per_emote.items():
        print(f"  {EMOTE_NAMES[emote_id]}: {count} samples")
    
    # Normalize embeddings
    print(f"\n🔧 Normalizing embeddings...")
    scaler = StandardScaler()
    embeddings = scaler.fit_transform(embeddings)
    
    # Split data (80% train, 20% validation)
    print(f"\n✂️  Splitting data (80% train, 20% val)...")
    X_train, X_val, y_train, y_val = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    
    # Create datasets
    train_dataset = PoseDataset(X_train, y_train)
    val_dataset = PoseDataset(X_val, y_val)
    
    # Create weighted sampler for balanced training
    sampler = create_weighted_sampler(y_train)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0
    )
    
    # Create model
    print(f"\n🏗️  Building Ultimate Model 4...")
    model = UltimateModel(input_dim=128, num_classes=7).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=10
    )
    
    # Training loop
    print(f"\n🎯 Training for {EPOCHS} epochs...")
    print("=" * 60)
    
    best_val_acc = 0
    patience_counter = 0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(EPOCHS):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Print progress
        print(f"Epoch {epoch+1:3d}/{EPOCHS} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_acc': train_acc,
                'val_acc': val_acc,
                'scaler_mean': scaler.mean_.tolist(),
                'scaler_scale': scaler.scale_.tolist(),
            }, MODEL_NAME)
            
            print(f"  ✅ New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= PATIENCE:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            break
    
    print("\n" + "=" * 60)
    print(f"🎉 Training complete!")
    print(f"🏆 Best validation accuracy: {best_val_acc:.2f}%")
    print("=" * 60)
    
    # Save model info
    model_info = {
        'model_name': 'Ultimate Model 4',
        'version': '4.0',
        'architecture': 'Deep Residual Network with Attention',
        'input_dim': 128,
        'num_classes': 7,
        'feature_type': 'MediaPipe 128-D Embeddings',
        'total_samples': len(embeddings),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'samples_per_emote': samples_per_emote,
        'best_val_accuracy': float(best_val_acc),
        'epochs_trained': epoch + 1,
        'batch_size': BATCH_SIZE,
        'learning_rate': LEARNING_RATE,
        'emote_names': EMOTE_NAMES,
        'layers': [
            'Input: 128-D embeddings',
            'FC1: 128 -> 512 (BatchNorm, ReLU, Dropout 0.4)',
            'FC2: 512 -> 512 (BatchNorm, ReLU, Dropout 0.4, Residual)',
            'FC3: 512 -> 256 (BatchNorm, ReLU, Dropout 0.3)',
            'FC4: 256 -> 256 (BatchNorm, ReLU, Dropout 0.3, Residual)',
            'FC5: 256 -> 128 (BatchNorm, ReLU, Dropout 0.2)',
            'Attention: 128 -> 128 (Softmax)',
            'Output: 128 -> 7 classes'
        ],
        'features': [
            'Residual connections',
            'Attention mechanism',
            'Batch normalization',
            'Dropout regularization',
            'Weighted sampling',
            'Learning rate scheduling',
            'Early stopping'
        ]
    }
    
    with open(INFO_NAME, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    print(f"\n💾 Model saved as: {MODEL_NAME}")
    print(f"📄 Info saved as: {INFO_NAME}")
    
    # Final statistics
    print(f"\n📊 Final Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Best validation accuracy: {best_val_acc:.2f}%")
    print(f"  Model size: {Path(MODEL_NAME).stat().st_size / 1024:.1f} KB")
    
    print(f"\n✅ Model 4 is ready to use!")
    print(f"   Select 'pose_model_4_ultimate.pth' from the dropdown in the web interface")

if __name__ == '__main__':
    main()
