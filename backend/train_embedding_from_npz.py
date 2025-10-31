"""
Train Embedding Model from NPZ Files
Loads individual .npz files, extracts 128-D embeddings, and trains the model
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
from .mediapipe_embedder import MediaPipePoseEmbedder
from .embedding_classifier import EmbeddingClassifierNet
from collections import Counter

print("=" * 70)
print("Training Embedding-Based Pose Classifier (Model 1.1)")
print("=" * 70)
print()

# Load data from pose_data_v2 directory (enhanced collector)
data_dir = "pose_data_v2"
print(f"📂 Loading training data from {data_dir}...")

# Fallback to old directory if new one doesn't exist
if not os.path.exists(data_dir):
    data_dir = "pose_data"
    print(f"   Using fallback directory: {data_dir}")

# Class mapping
class_names = [
    'crying',
    'laughing',
    'taunting',
    'yawning',
    'arms_folded_laughing',
    'hands_chest_kissing',
    'hands_raised_screaming'
]

class_to_idx = {name: idx for idx, name in enumerate(class_names)}

# Load all .npz files
all_landmarks = []
all_labels = []

for filename in os.listdir(data_dir):
    if filename.endswith('.npz'):
        # Extract class name from filename
        class_name = '_'.join(filename.split('_')[:-1])  # Remove the number part
        
        if class_name in class_to_idx:
            # Load the npz file
            filepath = os.path.join(data_dir, filename)
            data = np.load(filepath)
            
            # Get pose landmarks
            if 'pose' in data:
                landmarks = data['pose']
                all_landmarks.append(landmarks)
                all_labels.append(class_to_idx[class_name])

all_landmarks = np.array(all_landmarks)
all_labels = np.array(all_labels)

print(f"✅ Loaded {len(all_landmarks)} samples")
print(f"   Shape: {all_landmarks.shape}")
print()

# Check class distribution
label_counts = Counter(all_labels)
print("📊 Class distribution:")
for class_name, idx in class_to_idx.items():
    count = label_counts.get(idx, 0)
    print(f"   {class_name}: {count} samples")
print()

# Extract 128-D embeddings
print("🔄 Extracting 128-D MediaPipe embeddings...")
embedder = MediaPipePoseEmbedder()

embeddings = []
valid_labels = []

for i, landmarks in enumerate(all_landmarks):
    if i % 100 == 0:
        print(f"   Processing {i}/{len(all_landmarks)}...")
    
    try:
        embedding = embedder.extract_features(landmarks)
        embeddings.append(embedding)
        valid_labels.append(all_labels[i])
    except Exception as e:
        print(f"   ⚠️ Skipping sample {i}: {e}")
        continue

embeddings = np.array(embeddings)
valid_labels = np.array(valid_labels)

print(f"✅ Extracted embeddings: {embeddings.shape}")
print(f"   Valid samples: {len(valid_labels)}")
print()

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    embeddings, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
)

print(f"📊 Data split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")
print()

# Create datasets
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = torch.FloatTensor(embeddings)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

train_dataset = EmbeddingDataset(X_train, y_train)
test_dataset = EmbeddingDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Create model
num_classes = len(class_names)
model = EmbeddingClassifierNet(input_dim=128, num_classes=num_classes)

print(f"🧠 Model architecture:")
print(f"   Input: 128-D embeddings")
print(f"   Hidden: 256 -> 128 -> 64")
print(f"   Output: {num_classes} classes")
print()

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)

# Training loop
print("🚀 Starting training...")
print()

num_epochs = 50
best_accuracy = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for embeddings_batch, labels_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(embeddings_batch)
        loss = criterion(outputs, labels_batch)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels_batch.size(0)
        train_correct += (predicted == labels_batch).sum().item()
    
    train_accuracy = 100 * train_correct / train_total
    avg_train_loss = train_loss / len(train_loader)
    
    # Validation
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for embeddings_batch, labels_batch in test_loader:
            outputs = model(embeddings_batch)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels_batch.size(0)
            test_correct += (predicted == labels_batch).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    
    # Update learning rate
    scheduler.step(test_accuracy)
    
    # Print progress
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Train Acc: {train_accuracy:.2f}%")
        print(f"  Test Acc: {test_accuracy:.2f}%")
        print()
    
    # Save best model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_epoch = epoch + 1
        torch.save(model.state_dict(), "pose_embedding_classifier.pth")
        if (epoch + 1) % 5 != 0:
            print(f"  ✅ New best model at epoch {epoch+1} (Test Acc: {test_accuracy:.2f}%)")
            print()

print("=" * 70)
print("Training Complete!")
print("=" * 70)
print()
print(f"✅ Best Test Accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
print(f"✅ Model saved: pose_embedding_classifier.pth")
print()

# Save model info
info = {
    "input_dim": 128,
    "num_classes": num_classes,
    "class_names": class_names,
    "architecture": "128->256->128->64->7",
    "feature_type": "MediaPipe Pose Embeddings",
    "best_test_accuracy": float(best_accuracy),
    "best_epoch": best_epoch,
    "total_samples": len(all_landmarks),
    "training_samples": len(X_train),
    "test_samples": len(X_test),
    "status": "trained"
}

with open("pose_embedding_classifier_info.json", "w") as f:
    json.dump(info, f, indent=2)

print("✅ Model info saved: pose_embedding_classifier_info.json")
print()
print("Next steps:")
print("  1. Restart the web app (or it will auto-reload)")
print("  2. Select 'pose_embedding_classifier.pth' from the model dropdown")
print("  3. Enjoy MUCH better emote detection!")
print()
