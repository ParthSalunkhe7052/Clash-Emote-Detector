"""
PyTorch Neural Network Pose Classifier
For use in webapp - replaces RandomForest
"""

import torch
import torch.nn as nn
import numpy as np
import os


class PoseNeuralNet(nn.Module):
    """Neural Network Architecture (Model 1/2 - older architecture)"""
    def __init__(self, input_dim=18, num_classes=7):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


class Model3Classifier(nn.Module):
    """Model 3: Enhanced Deep Neural Network"""
    def __init__(self, input_dim=18, num_classes=7, dropout=0.35):
        super(Model3Classifier, self).__init__()
        
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layer 1
        self.hidden1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layer 2
        self.hidden2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Hidden layer 3
        self.hidden3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout * 0.7)
        )
        
        # Output layer
        self.output_layer = nn.Linear(64, num_classes)
    
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.hidden3(x)
        x = self.output_layer(x)
        return x


class NeuralPoseClassifier:
    """
    PyTorch-based pose classifier for webapp
    Much better than RandomForest!
    """
    def __init__(self, model_path='pose_neural_classifier.pth'):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        
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
        
        # Load model if exists
        if os.path.exists(model_path):
            self.load_model()
            print(f"✅ Neural model loaded from {model_path}")
        else:
            print(f"⚠️  Neural model not found: {model_path}")
            print(f"   Train the model first using train_neural_model.py or Google Colab")
    
    def load_model(self, input_dim=18, num_classes=7):
        """Load trained PyTorch model - auto-detects architecture"""
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            
            # Auto-detect architecture based on state_dict keys
            if 'input_layer.0.weight' in state_dict:
                # Model 3 architecture
                print(f"🔍 Detected Model 3 architecture")
                self.model = Model3Classifier(input_dim=input_dim, num_classes=num_classes)
            elif 'network.0.weight' in state_dict:
                # Old architecture (Model 1/2)
                print(f"🔍 Detected Model 1/2 architecture")
                self.model = PoseNeuralNet(input_dim=input_dim, num_classes=num_classes)
            else:
                raise ValueError(f"Unknown model architecture in {self.model_path}")
            
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading neural model: {e}")
            self.model = None
    
    def predict(self, features):
        """
        Predict pose from features
        
        Args:
            features: Feature vector (18-dim numpy array)
        
        Returns:
            pose_name: Predicted pose label
            confidence: Prediction confidence (0-1)
        """
        if self.model is None:
            return "No Model", 0.0
        
        try:
            # Convert to tensor
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            
            # Ensure correct shape
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            features = features.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            prediction = predicted.item()
            confidence = confidence.item()
            
            pose_name = self.pose_labels.get(prediction, "Unknown")
            
            return pose_name, confidence
        
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0
    
    def get_all_confidences(self, features):
        """
        Get confidence scores for all classes
        
        Args:
            features: Feature vector
        
        Returns:
            confidences: Dictionary of pose names and confidences
        """
        if self.model is None:
            return {}
        
        try:
            # Convert to tensor
            if isinstance(features, np.ndarray):
                features = torch.FloatTensor(features)
            
            if features.dim() == 1:
                features = features.unsqueeze(0)
            
            features = features.to(self.device)
            
            # Get probabilities
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)[0]
            
            # Convert to dict
            confidences = {}
            for i, prob in enumerate(probabilities):
                pose_name = self.pose_labels.get(i, f"Class_{i}")
                confidences[pose_name] = prob.item()
            
            return confidences
        
        except Exception as e:
            print(f"Error getting confidences: {e}")
            return {}


def extract_pose_features_simple(pose_landmarks):
    """
    Extract simple 18-dim features from pose landmarks
    Matches the training data format
    
    Args:
        pose_landmarks: MediaPipe pose landmarks (33x3)
    
    Returns:
        features: 18-dim feature vector
    """
    if pose_landmarks is None:
        return np.zeros(18)
    
    landmarks = np.array(pose_landmarks)
    
    # Key landmarks
    nose = landmarks[0]
    left_shoulder = landmarks[11]
    right_shoulder = landmarks[12]
    left_elbow = landmarks[13]
    right_elbow = landmarks[14]
    left_wrist = landmarks[15]
    right_wrist = landmarks[16]
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    features = []
    
    # 1-2: Shoulder positions (normalized by body center)
    body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
    body_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
    features.append(left_shoulder[0] - body_center_x)
    features.append(right_shoulder[0] - body_center_x)
    
    # 3-4: Wrist heights relative to shoulders
    features.append(left_shoulder[1] - left_wrist[1])
    features.append(right_shoulder[1] - right_wrist[1])
    
    # 5-6: Wrist horizontal positions
    features.append(left_wrist[0] - body_center_x)
    features.append(right_wrist[0] - body_center_x)
    
    # 7-8: Elbow positions
    features.append(left_elbow[0] - body_center_x)
    features.append(right_elbow[0] - body_center_x)
    
    # 9-10: Hip positions
    features.append(left_hip[0] - body_center_x)
    features.append(right_hip[0] - body_center_x)
    
    # 11: Nose position (head tilt)
    features.append(nose[0] - body_center_x)
    
    # 12-13: Arm lengths
    left_arm_len = np.linalg.norm(left_shoulder - left_wrist)
    right_arm_len = np.linalg.norm(right_shoulder - right_wrist)
    features.append(left_arm_len)
    features.append(right_arm_len)
    
    # 14-15: Shoulder width and hip width
    shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
    hip_width = np.linalg.norm(left_hip - right_hip)
    features.append(shoulder_width)
    features.append(hip_width)
    
    # 16: Torso height
    torso_height = abs(body_center_y - (left_hip[1] + right_hip[1]) / 2)
    features.append(torso_height)
    
    # 17: Body symmetry
    left_side = (left_shoulder + left_wrist + left_hip) / 3
    right_side = (right_shoulder + right_wrist + right_hip) / 3
    symmetry = np.linalg.norm(left_side - right_side)
    features.append(symmetry)
    
    # 18: Overall stance (avg y position of wrists)
    stance = (left_wrist[1] + right_wrist[1]) / 2
    features.append(stance)
    
    return np.array(features, dtype=np.float32)


if __name__ == "__main__":
    # Test the classifier
    print("Testing Neural Pose Classifier...")
    
    classifier = NeuralPoseClassifier()
    
    if classifier.model is not None:
        # Test with random features
        test_features = np.random.randn(18).astype(np.float32)
        pose, confidence = classifier.predict(test_features)
        
        print(f"\nTest Prediction:")
        print(f"  Pose: {pose}")
        print(f"  Confidence: {confidence:.2%}")
        
        # Get all confidences
        all_conf = classifier.get_all_confidences(test_features)
        print(f"\nAll Confidences:")
        for pose_name, conf in sorted(all_conf.items(), key=lambda x: x[1], reverse=True):
            print(f"  {pose_name}: {conf:.2%}")
    else:
        print("\n⚠️  Model not loaded. Train it first!")
