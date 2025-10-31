"""
Pose Classifier - Feature Extraction
Provides consistent feature extraction for training and inference
"""

import numpy as np


class PoseClassifier:
    """
    Feature extractor for pose landmarks
    Uses the same 18-dimensional feature extraction as neural classifier
    """
    
    def __init__(self):
        """Initialize classifier"""
        pass
    
    def extract_features(self, pose_landmarks):
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
        
        # Ensure we have the right shape
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
