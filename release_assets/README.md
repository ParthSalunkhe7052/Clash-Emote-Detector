# 📦 Release Assets - Model Files

This folder contains large pre-trained model files for the Clash Emote Detector v2.2.

---

## 🧠 Included Models

### 1. **Model 4 Ultimate** (Recommended)
- **File**: `pose_model_4_ultimate.pth`
- **Size**: ~7 MB
- **Architecture**: Advanced neural network with residual connections and attention
- **Features**: 128-D MediaPipe embeddings
- **Accuracy**: 95%+ on test set
- **Training**: 50 epochs, batch size 32, Adam optimizer
- **Best For**: Production use, highest accuracy

**Info File**: `pose_model_4_ultimate_info.json`

### 2. **Neural Classifier** (Enhanced)
- **File**: `pose_neural_classifier.pth`
- **Size**: ~196 KB
- **Architecture**: 3-layer feedforward network
- **Features**: 54-D enhanced visual features
- **Accuracy**: ~90%
- **Training**: 30 epochs, batch size 16
- **Best For**: Quick testing, lower memory footprint

**Info File**: `pose_neural_classifier_info.json`

### 3. **RandomForest Classifier** (Legacy)
- **File**: `pose_classifier_model_randomforest.pkl`
- **Size**: ~185 KB
- **Algorithm**: Traditional ML classifier
- **Features**: 18-D basic pose features
- **Accuracy**: ~75%
- **Best For**: Legacy support, very fast inference

---

## 📥 Installation

### Option 1: Download from GitHub Release
1. Download the release assets from the GitHub Releases page
2. Extract all `.pth` and `.pkl` files
3. Copy them to the `backend/models/` directory in your project

### Option 2: Copy from This Folder
If you have this repository cloned with all assets:
```bash
# Copy model files to backend/models
cp release_assets/*.pth backend/models/
cp release_assets/*.pkl backend/models/
cp release_assets/*.json backend/models/
```

---

## 🔄 Model File Structure

Each model requires:
- **Model weights file**: `.pth` or `.pkl`
- **Info file**: `_info.json` containing metadata
- **Label map**: `model_label_map.json` (shared across models)

### Label Map Format
```json
{
  "0": "Laughing",
  "1": "Yawning",
  "2": "Crying",
  "3": "Taunting",
  "4": "E Wiz",
  "5": "Kissing",
  "6": "Screaming"
}
```

---

## 🚀 Usage

Models are automatically loaded by the `unified_classifier.py` system:

```python
from backend.unified_classifier import UnifiedClassifier

# Initialize classifier with Model 4
classifier = UnifiedClassifier(model_type='model4')

# Switch models dynamically
classifier.switch_model('neural')
```

---

## 📊 Model Comparison

| Model | Size | Speed | Accuracy | Memory |
|-------|------|-------|----------|--------|
| Model 4 Ultimate | 7 MB | Medium (20 FPS) | 95%+ | ~500 MB |
| Neural Classifier | 196 KB | Fast (25 FPS) | 90% | ~300 MB |
| RandomForest | 185 KB | Very Fast (30 FPS) | 75% | ~200 MB |

---

## 🔧 Retraining Models

To retrain or fine-tune models with your own data:

```bash
# Activate virtual environment
call venv\Scripts\activate.bat

# Train Model 4 Ultimate
python backend\train_model_4_ultimate.py

# OR use the convenience script
retrain_model.bat
```

---

## 📝 Notes

- Model files are **optional** in the GitHub repository due to size
- They can be hosted on GitHub Releases or Git LFS
- Users can train their own models using the provided training scripts
- Pre-trained models are provided for convenience

---

## ⚠️ Important

- Do NOT commit these large files directly to the main repository
- Use Git LFS or GitHub Releases for distribution
- Users should download models separately if not included in clone

---

**Version**: 2.2.0  
**Last Updated**: October 31, 2025  
**Author**: Parth Salunkhe
