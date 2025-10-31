"""
Embedding-Based Pose Classifier
Uses 128-D MediaPipe embeddings with a simple neural network
Much better than the old 18-D features!
"""

import torch
import torch.nn as nn
import numpy as np
import os


class EmbeddingClassifierNet(nn.Module):
    """
    Neural network for classifying 128-D pose embeddings
    Simple but effective architecture with dropout for regularization
    """
    
    def __init__(self, input_dim=128, num_classes=7):
        super(EmbeddingClassifierNet, self).__init__()
        
        self.network = nn.Sequential(
            # Layer 1: 128 -> 256
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 2: 256 -> 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Layer 3: 128 -> 64
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class EmbeddingClassifier:
    """
    Wrapper for the embedding-based classifier
    Handles feature extraction, prediction, and model loading
    """
    
    def __init__(self, model_path=None, device='cpu'):
        """
        Initialize classifier
        
        Args:
            model_path: Path to trained model file (.pth)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.model = None
        self.classes = [
            'crying',
            'laughing', 
            'taunting',
            'yawning',
            'arms_folded_laughing',
            'hands_chest_kissing',
            'hands_raised_screaming'
        ]
        self.num_classes = len(self.classes)
        self.input_dim = 128
        
        # Load model if path provided
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        else:
            # Initialize untrained model
            self.model = EmbeddingClassifierNet(
                input_dim=self.input_dim,
                num_classes=self.num_classes
            ).to(self.device)
            self.model.eval()
    
    def load_model(self, model_path):
        """Load trained model from file"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Get classes if saved
                if 'classes' in checkpoint:
                    self.classes = checkpoint['classes']
                    self.num_classes = len(self.classes)
            else:
                state_dict = checkpoint
            
            # Create model with correct dimensions
            self.model = EmbeddingClassifierNet(
                input_dim=self.input_dim,
                num_classes=self.num_classes
            ).to(self.device)
            
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"✅ Loaded embedding classifier from {model_path}")
            print(f"   Input: {self.input_dim}-D embeddings")
            print(f"   Classes: {self.num_classes}")
            
        except Exception as e:
            print(f"⚠️ Error loading model: {e}")
            print(f"   Initializing untrained model")
            self.model = EmbeddingClassifierNet(
                input_dim=self.input_dim,
                num_classes=self.num_classes
            ).to(self.device)
            self.model.eval()
    
    def extract_features(self, pose_landmarks):
        """
        This is just for compatibility with the unified classifier
        The actual feature extraction happens in mediapipe_embedder.py
        """
        # This should not be called - embeddings are extracted by MediaPipePoseEmbedder
        raise NotImplementedError(
            "Use MediaPipePoseEmbedder.extract_features() instead"
        )
    
    def predict(self, embedding, landmark_data=None):
        """
        Predict pose class from embedding
        
        Args:
            embedding: 128-D pose embedding (numpy array)
            landmark_data: Not used, for compatibility
            
        Returns:
            (class_name, confidence)
        """
        if self.model is None:
            return "No Pose", 0.0
        
        # Convert to tensor
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        
        # Ensure correct shape
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        # Move to device
        embedding = embedding.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(embedding)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            class_idx = predicted.item()
            confidence_val = confidence.item()
            
            if class_idx < len(self.classes):
                class_name = self.classes[class_idx]
            else:
                class_name = "Unknown"
            
            return class_name, confidence_val
    
    def predict_proba(self, embedding):
        """
        Get probability distribution over all classes
        
        Args:
            embedding: 128-D pose embedding
            
        Returns:
            probabilities: Array of probabilities for each class
        """
        if self.model is None:
            return np.zeros(self.num_classes)
        
        # Convert to tensor
        if isinstance(embedding, np.ndarray):
            embedding = torch.FloatTensor(embedding)
        
        if embedding.dim() == 1:
            embedding = embedding.unsqueeze(0)
        
        embedding = embedding.to(self.device)
        
        with torch.no_grad():
            outputs = self.model(embedding)
            probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()[0]
    
    def get_classes(self):
        """Return list of class names"""
        return self.classes
    
    def get_input_dim(self):
        """Return input dimension"""
        return self.input_dim
