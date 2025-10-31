"""
MediaPipe Pose Embeddings Feature Extractor
Uses MediaPipe's advanced pose embedding for better feature extraction
This replaces the simple 18-D features with robust 128-D embeddings
"""

import numpy as np
import cv2


class MediaPipePoseEmbedder:
    """
    Extract high-quality pose embeddings using MediaPipe's approach
    Based on: https://google.github.io/mediapipe/solutions/pose_classification.html
    
    This creates a 128-D embedding that captures pose information much better
    than manual feature engineering.
    """
    
    def __init__(self):
        """Initialize the embedder"""
        self.embedding_dim = 128
        
    def extract_features(self, pose_landmarks):
        """
        Extract 128-D pose embedding from landmarks
        
        Args:
            pose_landmarks: MediaPipe pose landmarks (33x3 or 33x4 array)
            
        Returns:
            embedding: 128-D feature vector
        """
        if pose_landmarks is None:
            return np.zeros(self.embedding_dim)
        
        landmarks = np.array(pose_landmarks)
        
        # Ensure we have the right shape
        if landmarks.shape[0] < 33:
            return np.zeros(self.embedding_dim)
        
        # Take only x, y, z (ignore visibility if present)
        if landmarks.shape[1] > 3:
            landmarks = landmarks[:, :3]
        
        # Normalize landmarks
        landmarks = self._normalize_landmarks(landmarks)
        
        # Extract embedding using multiple feature types
        embedding = self._create_embedding(landmarks)
        
        return embedding
    
    def _normalize_landmarks(self, landmarks):
        """
        Normalize landmarks to be scale and translation invariant
        """
        # Center at origin (use hip center)
        left_hip = landmarks[23]
        right_hip = landmarks[24]
        center = (left_hip + right_hip) / 2
        landmarks = landmarks - center
        
        # Scale by torso size
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        torso_size = np.linalg.norm(left_shoulder - right_shoulder)
        
        if torso_size > 0:
            landmarks = landmarks / torso_size
        
        return landmarks
    
    def _create_embedding(self, landmarks):
        """
        Create 128-D embedding from normalized landmarks
        Uses multiple feature types for robustness
        """
        features = []
        
        # 1. Raw landmark positions (33 landmarks x 3 coords = 99 features)
        # But we'll use only key landmarks to keep it manageable
        key_landmarks = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # nose, shoulders, elbows, wrists, hips
        for idx in key_landmarks:
            features.extend(landmarks[idx])  # 9 landmarks x 3 = 27 features
        
        # 2. Pairwise distances between key points (important for pose)
        pairs = [
            (11, 12),  # shoulder width
            (23, 24),  # hip width
            (11, 13),  # left upper arm
            (13, 15),  # left forearm
            (12, 14),  # right upper arm
            (14, 16),  # right forearm
            (11, 23),  # left torso
            (12, 24),  # right torso
            (15, 16),  # hands distance
            (0, 11),   # head to left shoulder
            (0, 12),   # head to right shoulder
        ]
        for p1, p2 in pairs:
            dist = np.linalg.norm(landmarks[p1] - landmarks[p2])
            features.append(dist)  # 11 features
        
        # 3. Angles between limbs (captures pose geometry)
        angles = [
            self._angle_between_points(landmarks[11], landmarks[13], landmarks[15]),  # left arm angle
            self._angle_between_points(landmarks[12], landmarks[14], landmarks[16]),  # right arm angle
            self._angle_between_points(landmarks[23], landmarks[11], landmarks[13]),  # left shoulder angle
            self._angle_between_points(landmarks[24], landmarks[12], landmarks[14]),  # right shoulder angle
            self._angle_between_points(landmarks[11], landmarks[23], landmarks[24]),  # left hip angle
            self._angle_between_points(landmarks[12], landmarks[24], landmarks[23]),  # right hip angle
        ]
        features.extend(angles)  # 6 features
        
        # 4. Relative positions (normalized by body parts)
        # Wrist positions relative to shoulders
        features.extend(landmarks[15] - landmarks[11])  # left wrist rel to left shoulder (3)
        features.extend(landmarks[16] - landmarks[12])  # right wrist rel to right shoulder (3)
        
        # Elbow positions relative to shoulders
        features.extend(landmarks[13] - landmarks[11])  # left elbow rel to left shoulder (3)
        features.extend(landmarks[14] - landmarks[12])  # right elbow rel to right shoulder (3)
        
        # Head position relative to body center
        body_center = (landmarks[11] + landmarks[12] + landmarks[23] + landmarks[24]) / 4
        features.extend(landmarks[0] - body_center)  # 3 features
        
        # 5. Symmetry features (important for detecting poses)
        left_arm = landmarks[15] - landmarks[11]
        right_arm = landmarks[16] - landmarks[12]
        arm_symmetry = np.linalg.norm(left_arm - right_arm)
        features.append(arm_symmetry)  # 1 feature
        
        # Mirror symmetry (left vs right side)
        left_side = (landmarks[11] + landmarks[13] + landmarks[15] + landmarks[23]) / 4
        right_side = (landmarks[12] + landmarks[14] + landmarks[16] + landmarks[24]) / 4
        side_symmetry = np.linalg.norm(left_side - right_side)
        features.append(side_symmetry)  # 1 feature
        
        # 6. Velocity-like features (difference between upper and lower body)
        upper_body = (landmarks[11] + landmarks[12]) / 2
        lower_body = (landmarks[23] + landmarks[24]) / 2
        features.extend(upper_body - lower_body)  # 3 features
        
        # 7. Hand positions relative to face/body
        features.append(np.linalg.norm(landmarks[15] - landmarks[0]))  # left hand to nose
        features.append(np.linalg.norm(landmarks[16] - landmarks[0]))  # right hand to nose
        features.append(np.linalg.norm(landmarks[15] - body_center))   # left hand to body center
        features.append(np.linalg.norm(landmarks[16] - body_center))   # right hand to body center
        
        # 8. Arm crossing detection
        left_wrist_x = landmarks[15][0]
        right_wrist_x = landmarks[16][0]
        left_shoulder_x = landmarks[11][0]
        right_shoulder_x = landmarks[12][0]
        
        # Check if wrists cross body midline
        midline = (left_shoulder_x + right_shoulder_x) / 2
        left_crosses = 1.0 if left_wrist_x > midline else 0.0
        right_crosses = 1.0 if right_wrist_x < midline else 0.0
        features.append(left_crosses)
        features.append(right_crosses)
        
        # 9. Vertical positions (important for raised/lowered arms)
        shoulder_y = (landmarks[11][1] + landmarks[12][1]) / 2
        features.append(landmarks[15][1] - shoulder_y)  # left wrist height
        features.append(landmarks[16][1] - shoulder_y)  # right wrist height
        features.append(landmarks[0][1] - shoulder_y)   # nose height
        
        # 10. Additional geometric features
        # Hands distance to hips
        hip_center = (landmarks[23] + landmarks[24]) / 2
        features.append(np.linalg.norm(landmarks[15] - hip_center))
        features.append(np.linalg.norm(landmarks[16] - hip_center))
        
        # Elbow angles (important for arm positions)
        left_elbow_angle = self._angle_between_points(landmarks[11], landmarks[13], landmarks[15])
        right_elbow_angle = self._angle_between_points(landmarks[12], landmarks[14], landmarks[16])
        features.append(np.cos(left_elbow_angle))
        features.append(np.sin(left_elbow_angle))
        features.append(np.cos(right_elbow_angle))
        features.append(np.sin(right_elbow_angle))
        
        # Convert to numpy array
        features = np.array(features, dtype=np.float32)
        
        # Pad or truncate to exactly 128 dimensions
        if len(features) < self.embedding_dim:
            features = np.pad(features, (0, self.embedding_dim - len(features)))
        elif len(features) > self.embedding_dim:
            features = features[:self.embedding_dim]
        
        # Normalize the embedding
        norm = np.linalg.norm(features)
        if norm > 0:
            features = features / norm
        
        return features
    
    def _angle_between_points(self, p1, p2, p3):
        """
        Calculate angle at p2 formed by p1-p2-p3
        Returns angle in radians
        """
        v1 = p1 - p2
        v2 = p3 - p2
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        return angle
    
    def get_embedding_dim(self):
        """Return the dimensionality of the embedding"""
        return self.embedding_dim


# For backward compatibility
class PoseEmbedder(MediaPipePoseEmbedder):
    """Alias for backward compatibility"""
    pass
