"""Enhanced Visual Classifier Wrapper
Loads and runs inference with the enhanced 54-D feature model
"""

import torch
import torch.nn as nn
import numpy as np
import os


class EnhancedNeuralNet(nn.Module):
    """Enhanced Neural Network Architecture (matches training)"""
    def __init__(self, input_dim=54, num_classes=7, dropout=0.3):
        super(EnhancedNeuralNet, self).__init__()
        
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


class EnhancedVisualClassifier:
    """
    Enhanced visual emotion classifier using 54-D features
    Integrates pose, hand, and face landmarks
    """
    
    def __init__(self, model_path='enhanced_visual_classifier.pth'):
        """Initialize enhanced classifier"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EnhancedNeuralNet(input_dim=54, num_classes=7).to(self.device)
        
        # Load model weights
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Class labels
        self.pose_labels = {
            0: "Laughing",
            1: "Yawning",
            2: "Crying",
            3: "Taunting",
            4: "Arms Folded Laughing",
            5: "Hands Chest Kissing",
            6: "Hands Raised Screaming",
            7: "Unknown"
        }
    
    def predict(self, features):
        """
        Predict emote from 54-D features
        
        Args:
            features: numpy array of shape (54,) or (batch, 54)
        
        Returns:
            (emote_name, confidence)
        """
        # Convert to tensor
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        # Add batch dimension if needed
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        features = features.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get label
        label_idx = predicted.item()
        emote_name = self.pose_labels.get(label_idx, "Unknown")
        confidence_val = confidence.item()
        
        return emote_name, confidence_val
    
    def get_all_confidences(self, features):
        """
        Get confidence scores for all classes
        
        Args:
            features: numpy array of shape (54,)
        
        Returns:
            dict: {emote_name: confidence}
        """
        if isinstance(features, np.ndarray):
            features = torch.FloatTensor(features)
        
        if features.dim() == 1:
            features = features.unsqueeze(0)
        
        features = features.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features)
            probabilities = torch.softmax(outputs, dim=1)[0]
        
        # Create confidence dict
        confidences = {}
        for idx, prob in enumerate(probabilities):
            emote_name = self.pose_labels.get(idx, f"Class_{idx}")
            confidences[emote_name] = prob.item()
        
        return confidences


if __name__ == "__main__":
    # Test classifier
    print("Testing Enhanced Visual Classifier...")
    
    try:
        classifier = EnhancedVisualClassifier()
        
        # Test with random features
        test_features = np.random.randn(54).astype(np.float32)
        emote, confidence = classifier.predict(test_features)
        
        print(f"✅ Classifier loaded successfully")
        print(f"   Prediction: {emote} ({confidence*100:.1f}%)")
        
        # Get all confidences
        all_conf = classifier.get_all_confidences(test_features)
        print(f"\n   All confidences:")
        for name, conf in sorted(all_conf.items(), key=lambda x: x[1], reverse=True):
            print(f"      {name}: {conf*100:.1f}%")
    
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Train the model first: python training/train_enhanced_model.py")
