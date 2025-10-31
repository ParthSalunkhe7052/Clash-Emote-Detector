"""
Unified Classifier - Supports RandomForest, Neural Network, Embedding, and Model 4 Ultimate
"""

import numpy as np
import os
import pickle
import torch
import json
from pathlib import Path


class UnifiedClassifier:
    """
    Unified classifier that can load and use:
    - RandomForest models (.pkl) - Original version
    - PyTorch Neural Network models (.pth) - 18-D or 54-D
    - Embedding-based models (.pth) - 128-D MediaPipe embeddings (NEW!)
    """
    
    def __init__(self, model_path=None):
        """Initialize classifier with model"""
        self.model = None
        self.model_type = None
        self.model_path = model_path
        self.input_dim = 18  # Default to 18-D for backward compatibility
        self.feature_extractor = None  # Will be set based on model type
        self.pose_labels = {
            0: "Laughing",
            1: "Yawning",
            2: "Crying",
            3: "Taunting",
            4: "E Wiz",
            5: "Kissing",
            6: "Screaming",
            7: "Unknown"
        }
        
        # Load label mapping for backward compatibility
        self.label_map = None
        try:
            import json
            map_path = Path(__file__).parent / 'model_label_map.json'
            if map_path.exists():
                with open(map_path, 'r') as f:
                    label_map_data = json.load(f)
                    self.label_map = label_map_data.get('model_to_canonical', {})
        except Exception as e:
            pass
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load model from file (auto-detects type)"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model_path = model_path
        
        # Detect model type by extension
        if model_path.endswith('.pkl'):
            # RandomForest model
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            self.model_type = 'randomforest'
            print(f"✅ Loaded RandomForest model: {model_path}")
            
        elif model_path.endswith('.pth'):
            # PyTorch Neural Network - detect if 18-D or 54-D
            # Check for model info file
            info_file = model_path.replace('.pth', '_info.json')
            if not os.path.exists(info_file):
                # Try alternate names
                if 'enhanced' in model_path.lower():
                    info_file = 'enhanced_model_info.json'
                else:
                    info_file = 'model_info.json'
            
            # Determine input dimension
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    info = json.load(f)
                    self.input_dim = info.get('input_dim', 18)
            else:
                # Infer from model name
                if 'embedding' in model_path.lower():
                    self.input_dim = 128
                elif 'enhanced' in model_path.lower():
                    self.input_dim = 54
                else:
                    self.input_dim = 18
            
            # Load appropriate model architecture
            if self.input_dim == 128:
                # Check if this is Model 4 Ultimate
                if 'model_4' in model_path.lower() or 'ultimate' in model_path.lower():
                    # Model 4 Ultimate
                    from backend.model_4_classifier import Model4Classifier
                    
                    self.model = Model4Classifier(model_path=model_path, info_path=info_file)
                    self.feature_extractor = self.model.embedder
                    self.model_type = 'model4'
                    print(f"✅ Loaded Model 4 Ultimate (128-D): {model_path}")
                    print(f"   🚀 Advanced architecture with residual connections & attention!")
                else:
                    # Regular embedding model
                    from backend.embedding_classifier import EmbeddingClassifier
                    from backend.mediapipe_embedder import MediaPipePoseEmbedder
                    
                    self.model = EmbeddingClassifier(model_path=model_path)
                    self.feature_extractor = MediaPipePoseEmbedder()
                    self.model_type = 'embedding'
                    print(f"✅ Loaded Embedding-Based Neural Network (128-D): {model_path}")
                    print(f"   Using MediaPipe pose embeddings for better accuracy!")
                
            elif self.input_dim == 54:
                from backend.enhanced_classifier import EnhancedNeuralNet
                from backend.enhanced_visual_features import EnhancedVisualFeatureExtractor
                self.model = EnhancedNeuralNet(input_dim=54, num_classes=7)
                self.feature_extractor = EnhancedVisualFeatureExtractor()
                print(f"✅ Loaded Enhanced Neural Network (54-D): {model_path}")
                
                # Load weights
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                self.model.load_state_dict(state_dict)
                self.model.eval()
            else:
                # Auto-detect architecture for 18-D models
                from backend.pose_neural_classifier import PoseNeuralNet, Model3Classifier
                
                # Load state dict first to detect architecture
                state_dict = torch.load(model_path, map_location=torch.device('cpu'))
                
                if 'input_layer.0.weight' in state_dict:
                    # Model 3 architecture
                    print(f"🔍 Detected Model 3 architecture")
                    self.model = Model3Classifier(input_dim=18, num_classes=7)
                elif 'network.0.weight' in state_dict:
                    # Old architecture (Model 1/2)
                    print(f"🔍 Detected Model 1/2 architecture")
                    self.model = PoseNeuralNet(input_dim=18, num_classes=7)
                else:
                    raise ValueError(f"Unknown model architecture in {model_path}")
                
                print(f"✅ Loaded Neural Network (18-D): {model_path}")
                
                # Load weights
                self.model.load_state_dict(state_dict)
                self.model.eval()
            
            # Only set model_type to 'neural' if not already set (e.g., for Model 4)
            if self.model_type is None:
                self.model_type = 'neural'
        else:
            raise ValueError(f"Unsupported model file type: {model_path}")
    
    def extract_features(self, pose_landmarks):
        """Extract 18-dim features from pose landmarks"""
        if pose_landmarks is None:
            return np.zeros(18)
        
        landmarks = np.array(pose_landmarks)
        
        if landmarks.shape[0] < 33:
            return np.zeros(18)
        
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
        
        # Body center
        body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        body_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # 1-2: Shoulder positions
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
        
        # 11: Nose position
        features.append(nose[0] - body_center_x)
        
        # 12-13: Arm lengths
        left_arm_len = np.linalg.norm(left_shoulder - left_wrist)
        right_arm_len = np.linalg.norm(right_shoulder - right_wrist)
        features.append(left_arm_len)
        features.append(right_arm_len)
        
        # 14-15: Widths
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        hip_width = np.linalg.norm(left_hip - right_hip)
        features.append(shoulder_width)
        features.append(hip_width)
        
        # 16: Torso height
        torso_height = abs(body_center_y - (left_hip[1] + right_hip[1]) / 2)
        features.append(torso_height)
        
        # 17: Symmetry
        left_side = (left_shoulder + left_wrist + left_hip) / 3
        right_side = (right_shoulder + right_wrist + right_hip) / 3
        symmetry = np.linalg.norm(left_side - right_side)
        features.append(symmetry)
        
        # 18: Stance
        stance = (left_wrist[1] + right_wrist[1]) / 2
        features.append(stance)
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, features, landmark_data=None):
        """
        Predict emote from features
        
        Args:
            features: Feature vector (18-D, 54-D, or 128-D) or pose landmarks
            landmark_data: Optional dict with all landmarks for enhanced features
        
        Returns:
            pose_name: Predicted emote name
            confidence: Confidence score (0-1)
        """
        if self.model is None:
            return "No Model", 0.0
        
        try:
            # For Model 4 Ultimate (128-D with advanced architecture)
            if self.model_type == 'model4':
                # Model 4 needs raw landmarks (33, 4)
                # Features should already be landmarks from MediaPipe
                if len(features.shape) == 1:
                    # 1D array - could be flattened landmarks or extracted features
                    if len(features) == 132:  # 33 * 4 = 132 (flattened landmarks)
                        landmarks = features.reshape(33, 4)
                    else:
                        # Invalid - can't use extracted features with Model 4
                        return "Invalid Features", 0.0
                else:
                    # Already 2D - should be (33, 4)
                    landmarks = features
                
                # Validate shape
                if landmarks.shape != (33, 4):
                    return "Invalid Features", 0.0
                
                # Model 4's predict returns (class_id, confidence, probabilities)
                # But we need to get the emote name from it
                class_id, confidence, probs = self.model.predict(landmarks)
                pose_name = self.model.get_emote_name(class_id)
                
                # Apply label mapping to convert old labels to new ones
                if self.label_map and pose_name in self.label_map:
                    canonical_id = self.label_map[pose_name]
                    # Load manifest to get display label
                    try:
                        manifest_path = Path(__file__).parent / 'emotes' / 'manifest.json'
                        if manifest_path.exists():
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                                for emote in manifest['emotes']:
                                    if emote['id'] == canonical_id:
                                        pose_name = emote['label']
                                        break
                    except Exception as e:
                        pass  # If mapping fails, use original label
                
                return pose_name, float(confidence)
            
            # For embedding models
            if self.model_type == 'embedding':
                # Extract embedding features if needed
                if landmark_data is not None and 'pose' in landmark_data:
                    embedding_features = self.feature_extractor.extract_features(landmark_data['pose'])
                else:
                    embedding_features = self.feature_extractor.extract_features(features)
                
                # Use embedding classifier's predict method
                pose_name, confidence = self.model.predict(embedding_features, landmark_data)
                
                # Apply label mapping to convert old labels to new ones
                if self.label_map and pose_name in self.label_map:
                    canonical_id = self.label_map[pose_name]
                    # Load manifest to get display label
                    try:
                        manifest_path = Path(__file__).parent / 'emotes' / 'manifest.json'
                        if manifest_path.exists():
                            with open(manifest_path, 'r') as f:
                                manifest = json.load(f)
                                for emote in manifest['emotes']:
                                    if emote['id'] == canonical_id:
                                        pose_name = emote['label']
                                        break
                    except Exception as e:
                        pass  # If mapping fails, use original label
                
                return pose_name, confidence
            
            # For enhanced 54-D models, use landmark_data if provided
            elif self.input_dim == 54 and landmark_data is not None:
                features = self.feature_extractor.extract_features(landmark_data)
            # If features are landmarks, extract features first
            elif len(features.shape) > 1 or len(features) == 33:
                features = self.extract_features(features)
            
            # Ensure features match expected dimension
            if len(features) != self.input_dim:
                return "Invalid Features", 0.0
            
            if self.model_type == 'randomforest':
                # RandomForest prediction
                features_2d = features.reshape(1, -1)
                prediction = self.model.predict(features_2d)[0]
                
                # Get probability for confidence
                try:
                    probabilities = self.model.predict_proba(features_2d)[0]
                    confidence = float(np.max(probabilities))
                except:
                    confidence = 0.7  # Default confidence for RF
                
                pose_name = self.pose_labels.get(prediction, "Unknown")
                
            elif self.model_type == 'neural':
                # Neural Network prediction
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = self.model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                
                prediction = predicted.item()
                confidence = confidence.item()
                pose_name = self.pose_labels.get(prediction, "Unknown")
            
            else:
                return "Unknown Model Type", 0.0
            
            # Map to canonical label if label map is available
            if self.label_map and pose_name in self.label_map:
                canonical_id = self.label_map[pose_name]
                # Load manifest to get display label
                try:
                    manifest_path = Path(__file__).parent / 'emotes' / 'manifest.json'
                    if manifest_path.exists():
                        with open(manifest_path, 'r') as f:
                            manifest = json.load(f)
                            for emote in manifest['emotes']:
                                if emote['id'] == canonical_id:
                                    pose_name = emote['label']
                                    break
                except:
                    pass
            
            return pose_name, float(confidence)
            
        except Exception as e:
            print(f"Prediction error: {e}")
            import traceback
            traceback.print_exc()
            return "Error", 0.0
    
    def get_model_info(self):
        """Get information about current model"""
        if self.model is None:
            return {
                'type': 'None',
                'path': None,
                'loaded': False
            }
        
        return {
            'type': self.model_type,
            'path': self.model_path,
            'loaded': True,
            'name': os.path.basename(self.model_path) if self.model_path else 'Unknown'
        }
