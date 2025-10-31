"""
Model 4 Ultimate Classifier
Loads and uses the ultimate Model 4 for predictions
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from .mediapipe_embedder import MediaPipePoseEmbedder

class UltimateModel(nn.Module):
    """
    Ultimate neural network architecture for Model 4
    Must match the architecture in training script
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

class Model4Classifier:
    """Classifier for Model 4 Ultimate"""
    
    def __init__(self, model_path='pose_model_4_ultimate.pth', info_path='pose_model_4_ultimate_info.json'):
        """Initialize the classifier"""
        self.model_path = Path(model_path)
        self.info_path = Path(info_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model info
        with open(self.info_path, 'r') as f:
            self.info = json.load(f)
        
        self.emote_names = self.info['emote_names']
        self.num_classes = self.info['num_classes']
        
        # Initialize embedder
        self.embedder = MediaPipePoseEmbedder()
        
        # Load model
        self.model = UltimateModel(input_dim=128, num_classes=self.num_classes).to(self.device)
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Load scaler parameters
        self.scaler_mean = np.array(checkpoint['scaler_mean'])
        self.scaler_scale = np.array(checkpoint['scaler_scale'])
        
        print(f"✅ Model 4 Ultimate loaded")
        print(f"   Validation accuracy: {self.info['best_val_accuracy']:.2f}%")
    
    def predict(self, landmarks):
        """
        Predict emote from pose landmarks
        
        Args:
            landmarks: numpy array of shape (33, 4) with x, y, z, visibility
            
        Returns:
            tuple: (predicted_class_id, confidence, class_probabilities)
        """
        # Extract 128-D embedding
        embedding = self.embedder.extract_features(landmarks)
        
        # Normalize using scaler
        embedding = (embedding - self.scaler_mean) / self.scaler_scale
        
        # Convert to tensor
        embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(embedding_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = predicted.item()
        confidence_score = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence_score, all_probs
    
    def get_emote_name(self, class_id):
        """Get emote name from class ID"""
        return self.emote_names[str(class_id)]
    
    def get_model_info(self):
        """Get model information"""
        return self.info
