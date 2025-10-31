import numpy as np
import os

print("Loading all collected samples...")

all_features = []
all_labels = []

# Load all .npz files
count = 0
for filename in os.listdir('pose_data'):
    if filename.endswith('.npz'):
        filepath = os.path.join('pose_data', filename)
        data = np.load(filepath)
        all_features.append(data['features'])
        all_labels.append(data['label'])
        count += 1

print(f"Loaded {count} samples")

# Convert to arrays
X = np.array(all_features)
y = np.array(all_labels)

# Save
np.save('pose_data/pose_features_latest.npy', X)
np.save('pose_data/pose_labels_latest.npy', y)

# Show distribution
unique, counts = np.unique(y, return_counts=True)
emote_names = {
    0: "Laughing",
    1: "Yawning", 
    2: "Crying",
    3: "Taunting",
    4: "Arms Folded Laughing",
    5: "Hands Chest Kissing",
    6: "Hands Raised Screaming"
}

print(f"\nData Distribution:")
for label, count in zip(unique, counts):
    print(f"  {emote_names[label]}: {count} samples")

print(f"\nTotal: {len(X)} samples")
print(f"Feature dimension: {X.shape[1]}")
print(f"\nExported to pose_data/pose_features_latest.npy and pose_data/pose_labels_latest.npy")
