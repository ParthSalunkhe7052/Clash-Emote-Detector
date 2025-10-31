"""
Create Initial Embedding Model
Creates an untrained 128-D embedding-based model as Model 1.1
"""

import torch
from .embedding_classifier import EmbeddingClassifierNet

print("=" * 70)
print("Creating Initial Embedding Model (Model 1.1)")
print("=" * 70)
print()

# Create model
model = EmbeddingClassifierNet(input_dim=128, num_classes=7)

# Save model
model_path = "pose_embedding_classifier.pth"
torch.save(model.state_dict(), model_path)

print(f"✅ Created untrained embedding model: {model_path}")
print()
print("Model Details:")
print(f"  - Input: 128-D MediaPipe embeddings")
print(f"  - Architecture: 128 -> 256 -> 128 -> 64 -> 7")
print(f"  - Classes: 7 emotes")
print(f"  - Status: UNTRAINED (needs training data)")
print()
print("Next Steps:")
print("  1. Collect training data using the web interface")
print("  2. Train the model using train_embedding_model.py")
print("  3. Model will be MUCH better than the old 18-D model!")
print()
