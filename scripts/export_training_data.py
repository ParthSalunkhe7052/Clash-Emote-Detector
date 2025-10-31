"""Export Training Data for Google Colab

Author: Parth
Exports collected training data to a format ready for Colab training.
"""

import os
import json
import numpy as np
import pickle
from datetime import datetime


def export_pose_data(data_dir='pose_data', output_dir='training_export'):
    """
    Export pose training data for Google Colab
    
    Args:
        data_dir: Directory containing collected pose data
        output_dir: Directory to save exported data
    """
    print("=" * 50)
    print("Clash Emote Detector - Training Data Exporter")
    print("=" * 50)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(data_dir):
        print(f"❌ Error: {data_dir} not found!")
        print("Please collect training data first using data_collector.py")
        return False
    
    # Load data
    features_file = os.path.join(data_dir, 'pose_features_latest.npy')
    labels_file = os.path.join(data_dir, 'pose_labels_latest.npy')
    metadata_file = os.path.join(data_dir, 'pose_metadata_latest.json')
    
    if not os.path.exists(features_file):
        print(f"❌ No training data found in {data_dir}")
        print("Collect data first!")
        return False
    
    # Load features and labels
    try:
        X = np.load(features_file)
        y = np.load(labels_file)
        
        print(f"\n✅ Loaded training data:")
        print(f"   Features shape: {X.shape}")
        print(f"   Labels shape: {y.shape}")
        print(f"   Total samples: {len(X)}")
        
        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        class_names = {0: "Laughing", 1: "Yawning", 2: "Crying", 3: "Taunting"}
        
        print(f"\n📊 Samples per emote:")
        for label, count in zip(unique, counts):
            print(f"   {class_names.get(label, f'Class {label}')}: {count} samples")
        
        # Load metadata if exists
        metadata = {}
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Save to output directory
        output_features = os.path.join(output_dir, 'pose_features.npy')
        output_labels = os.path.join(output_dir, 'pose_labels.npy')
        output_metadata = os.path.join(output_dir, 'metadata.json')
        
        np.save(output_features, X)
        np.save(output_labels, y)
        
        # Create metadata
        export_metadata = {
            'export_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_samples': int(len(X)),
            'feature_dim': int(X.shape[1]),
            'num_classes': int(len(unique)),
            'class_distribution': {
                class_names.get(label, f'Class {label}'): int(count)
                for label, count in zip(unique, counts)
            },
            'class_mapping': class_names,
            'original_metadata': metadata
        }
        
        with open(output_metadata, 'w') as f:
            json.dump(export_metadata, f, indent=2)
        
        print(f"\n✅ Data exported to: {output_dir}/")
        print(f"   - pose_features.npy ({X.nbytes / 1024:.2f} KB)")
        print(f"   - pose_labels.npy ({y.nbytes / 1024:.2f} KB)")
        print(f"   - metadata.json")
        
        # Create README
        readme_path = os.path.join(output_dir, 'README.txt')
        with open(readme_path, 'w') as f:
            f.write("Clash Emote Detector - Training Data\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Exported: {export_metadata['export_date']}\n")
            f.write(f"Total Samples: {export_metadata['total_samples']}\n")
            f.write(f"Feature Dimension: {export_metadata['feature_dim']}\n\n")
            f.write("Class Distribution:\n")
            for cls, count in export_metadata['class_distribution'].items():
                f.write(f"  {cls}: {count} samples\n")
            f.write("\n" + "=" * 50 + "\n")
            f.write("Next Steps:\n")
            f.write("1. Compress this folder to .zip\n")
            f.write("2. Upload to Google Drive\n")
            f.write("3. Use TRAINING_GUIDE.md for instructions\n")
            f.write("4. Train model in Google Colab\n")
            f.write("5. Download trained model\n")
            f.write("6. Import back to project\n")
        
        print(f"   - README.txt")
        
        # Compress if possible
        try:
            import zipfile
            zip_path = f'{output_dir}.zip'
            with zipfile.ZipFile(zip_path, 'w') as zipf:
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(output_dir))
                        zipf.write(file_path, arcname)
            
            print(f"\n✅ Created zip file: {zip_path}")
            print(f"   Upload this to Google Drive!")
            
        except Exception as e:
            print(f"\n⚠️ Could not create zip file: {e}")
            print(f"   Manually compress the '{output_dir}' folder")
        
        print("\n" + "=" * 50)
        print("✅ Export complete!")
        print(f"📚 See TRAINING_GUIDE.md for next steps")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False


if __name__ == "__main__":
    print("\n🚀 Starting data export...\n")
    success = export_pose_data()
    
    if success:
        print("\n✅ Ready for Google Colab training!")
    else:
        print("\n❌ Export failed. Check errors above.")
