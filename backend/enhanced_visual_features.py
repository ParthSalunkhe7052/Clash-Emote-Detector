"""Enhanced Visual Feature Extractor for Emotion Detection

Extracts comprehensive features from:
- Pose landmarks (body position, arm angles)
- Hand landmarks (gestures, positions relative to face/body)
- Face landmarks (mouth openness, eye state, expression)

This provides much richer features than pose-only for better emotion classification.
"""

import numpy as np


class EnhancedVisualFeatureExtractor:
    """
    Extract rich visual features from MediaPipe Holistic landmarks
    
    Feature dimensions:
    - Pose features: 18 (existing)
    - Hand features: 24 (12 per hand - position, openness, gestures)
    - Face features: 12 (mouth, eyes, expression)
    Total: 54 dimensions
    """
    
    def __init__(self):
        """Initialize feature extractor"""
        pass
    
    def extract_pose_features(self, pose_landmarks):
        """
        Extract 18-D pose features (same as before for compatibility)
        
        Args:
            pose_landmarks: MediaPipe pose landmarks (33x3)
        
        Returns:
            features: 18-dim feature vector
        """
        if pose_landmarks is None or len(pose_landmarks) < 33:
            return np.zeros(18, dtype=np.float32)
        
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
        
        # Body center
        body_center_x = (left_shoulder[0] + right_shoulder[0]) / 2
        body_center_y = (left_shoulder[1] + right_shoulder[1]) / 2
        
        # 1-2: Shoulder positions (normalized by body center)
        features.append(left_shoulder[0] - body_center_x)
        features.append(right_shoulder[0] - body_center_x)
        
        # 3-4: Wrist heights relative to shoulders
        features.append(left_shoulder[1] - left_wrist[1])
        features.append(right_shoulder[1] - right_wrist[1])
        
        # 5-6: Wrist horizontal positions
        features.append(left_wrist[0] - body_center_x)
        features.append(right_wrist[0] - body_center_x)
        
        # 7-8: Elbow angles (approximate)
        left_arm_angle = np.arctan2(left_wrist[1] - left_elbow[1], left_wrist[0] - left_elbow[0])
        right_arm_angle = np.arctan2(right_wrist[1] - right_elbow[1], right_wrist[0] - right_elbow[0])
        features.append(left_arm_angle)
        features.append(right_arm_angle)
        
        # 9-10: Shoulder-elbow distances
        left_upper_arm = np.linalg.norm(left_shoulder - left_elbow)
        right_upper_arm = np.linalg.norm(right_shoulder - right_elbow)
        features.append(left_upper_arm)
        features.append(right_upper_arm)
        
        # 11-12: Elbow-wrist distances
        left_forearm = np.linalg.norm(left_elbow - left_wrist)
        right_forearm = np.linalg.norm(right_elbow - right_wrist)
        features.append(left_forearm)
        features.append(right_forearm)
        
        # 13-14: Hip positions
        features.append(left_hip[0] - body_center_x)
        features.append(right_hip[0] - body_center_x)
        
        # 15: Torso tilt
        torso_tilt = np.arctan2(body_center_y - (left_hip[1] + right_hip[1])/2, 
                                body_center_x - (left_hip[0] + right_hip[0])/2)
        features.append(torso_tilt)
        
        # 16-17: Nose position relative to body
        features.append(nose[0] - body_center_x)
        features.append(nose[1] - body_center_y)
        
        # 18: Shoulder width
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        features.append(shoulder_width)
        
        return np.array(features, dtype=np.float32)
    
    def extract_hand_features(self, left_hand, right_hand, pose_landmarks):
        """
        Extract 24-D hand features (12 per hand)
        
        Features per hand:
        - Position relative to face (x, y)
        - Position relative to body center (x, y)
        - Hand openness (finger spread)
        - Hand height relative to shoulder
        - Distance from face
        - Wrist-to-fingertip distances (thumb, index, pinky)
        - Hand orientation
        
        Args:
            left_hand: Left hand landmarks (21x3) or None
            right_hand: Right hand landmarks (21x3) or None
            pose_landmarks: Pose landmarks for reference points
        
        Returns:
            features: 24-dim feature vector
        """
        features = []
        
        # Get reference points from pose
        if pose_landmarks is not None and len(pose_landmarks) >= 33:
            nose = pose_landmarks[0]
            left_shoulder = pose_landmarks[11]
            right_shoulder = pose_landmarks[12]
            body_center = np.array([(left_shoulder[0] + right_shoulder[0]) / 2,
                                   (left_shoulder[1] + right_shoulder[1]) / 2])
        else:
            nose = np.array([0.5, 0.3, 0])
            body_center = np.array([0.5, 0.5])
            left_shoulder = np.array([0.4, 0.4, 0])
            right_shoulder = np.array([0.6, 0.4, 0])
        
        # Process each hand
        for hand_landmarks, shoulder in [(left_hand, left_shoulder), (right_hand, right_shoulder)]:
            if hand_landmarks is None or len(hand_landmarks) < 21:
                # No hand detected - add zeros
                features.extend([0.0] * 12)
            else:
                hand = np.array(hand_landmarks)
                wrist = hand[0]
                thumb_tip = hand[4]
                index_tip = hand[8]
                middle_tip = hand[12]
                ring_tip = hand[16]
                pinky_tip = hand[20]
                
                # 1-2: Position relative to face
                features.append(wrist[0] - nose[0])
                features.append(wrist[1] - nose[1])
                
                # 3-4: Position relative to body center
                features.append(wrist[0] - body_center[0])
                features.append(wrist[1] - body_center[1])
                
                # 5: Hand openness (average fingertip spread from wrist)
                fingertips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
                openness = np.mean([np.linalg.norm(tip - wrist) for tip in fingertips])
                features.append(openness)
                
                # 6: Hand height relative to shoulder
                features.append(shoulder[1] - wrist[1])
                
                # 7: Distance from face (proximity to face)
                face_dist = np.linalg.norm(wrist[:2] - nose[:2])
                features.append(face_dist)
                
                # 8-10: Wrist-to-fingertip distances (thumb, index, pinky)
                features.append(np.linalg.norm(thumb_tip - wrist))
                features.append(np.linalg.norm(index_tip - wrist))
                features.append(np.linalg.norm(pinky_tip - wrist))
                
                # 11: Hand orientation (angle of index finger)
                index_angle = np.arctan2(index_tip[1] - wrist[1], index_tip[0] - wrist[0])
                features.append(index_angle)
                
                # 12: Finger curl (how much fingers are bent)
                # Compare distance of middle fingertip to wrist vs middle knuckle to wrist
                middle_knuckle = hand[9]  # Middle finger MCP joint
                curl_ratio = np.linalg.norm(middle_tip - wrist) / (np.linalg.norm(middle_knuckle - wrist) + 1e-6)
                features.append(curl_ratio)
        
        return np.array(features, dtype=np.float32)
    
    def extract_face_features(self, face_landmarks):
        """
        Extract 12-D face features for expression detection
        
        Features:
        - Mouth openness (vertical)
        - Mouth width
        - Lip corners position (smile/frown)
        - Eye openness (left, right)
        - Eyebrow position (left, right)
        - Face tilt
        - Jaw openness
        
        Args:
            face_landmarks: MediaPipe face landmarks (468x3) or None
        
        Returns:
            features: 12-dim feature vector
        """
        if face_landmarks is None or len(face_landmarks) < 468:
            return np.zeros(12, dtype=np.float32)
        
        face = np.array(face_landmarks)
        
        features = []
        
        # Key facial landmarks (MediaPipe face mesh indices)
        # Mouth
        upper_lip = face[13]  # Upper lip center
        lower_lip = face[14]  # Lower lip center
        left_mouth = face[61]  # Left mouth corner
        right_mouth = face[291]  # Right mouth corner
        
        # Eyes
        left_eye_top = face[159]
        left_eye_bottom = face[145]
        right_eye_top = face[386]
        right_eye_bottom = face[374]
        
        # Eyebrows
        left_eyebrow = face[70]
        right_eyebrow = face[300]
        
        # Nose and face reference
        nose_tip = face[1]
        chin = face[152]
        
        # 1: Mouth openness (vertical distance)
        mouth_open = np.linalg.norm(upper_lip - lower_lip)
        features.append(mouth_open)
        
        # 2: Mouth width
        mouth_width = np.linalg.norm(left_mouth - right_mouth)
        features.append(mouth_width)
        
        # 3-4: Lip corner positions (smile/frown detection)
        # Higher Y = lower on screen (frown), Lower Y = higher (smile)
        features.append(left_mouth[1] - upper_lip[1])
        features.append(right_mouth[1] - upper_lip[1])
        
        # 5-6: Eye openness
        left_eye_open = np.linalg.norm(left_eye_top - left_eye_bottom)
        right_eye_open = np.linalg.norm(right_eye_top - right_eye_bottom)
        features.append(left_eye_open)
        features.append(right_eye_open)
        
        # 7-8: Eyebrow height (relative to eyes)
        left_brow_height = left_eye_top[1] - left_eyebrow[1]
        right_brow_height = right_eye_top[1] - right_eyebrow[1]
        features.append(left_brow_height)
        features.append(right_brow_height)
        
        # 9: Face tilt (angle)
        face_tilt = np.arctan2(right_mouth[1] - left_mouth[1], right_mouth[0] - left_mouth[0])
        features.append(face_tilt)
        
        # 10: Jaw openness (chin to nose distance)
        jaw_open = np.linalg.norm(chin - nose_tip)
        features.append(jaw_open)
        
        # 11: Mouth aspect ratio (openness relative to width)
        mouth_aspect = mouth_open / (mouth_width + 1e-6)
        features.append(mouth_aspect)
        
        # 12: Smile intensity (lip corners relative to mouth center)
        smile_intensity = (left_mouth[1] + right_mouth[1]) / 2 - (upper_lip[1] + lower_lip[1]) / 2
        features.append(smile_intensity)
        
        return np.array(features, dtype=np.float32)
    
    def extract_features(self, landmark_data):
        """
        Extract all enhanced visual features
        
        Args:
            landmark_data: Dict with keys 'pose', 'left_hand', 'right_hand', 'face'
                          (output from HolisticDetector.get_landmark_data)
        
        Returns:
            features: 54-dim feature vector (18 pose + 24 hands + 12 face)
        """
        pose = landmark_data.get('pose')
        left_hand = landmark_data.get('left_hand')
        right_hand = landmark_data.get('right_hand')
        face = landmark_data.get('face')
        
        pose_feat = self.extract_pose_features(pose)
        hand_feat = self.extract_hand_features(left_hand, right_hand, pose)
        face_feat = self.extract_face_features(face)
        
        # Concatenate all features
        all_features = np.concatenate([pose_feat, hand_feat, face_feat])
        
        return all_features.astype(np.float32)


if __name__ == "__main__":
    # Test with dummy data
    extractor = EnhancedVisualFeatureExtractor()
    
    dummy_data = {
        'pose': np.random.randn(33, 3),
        'left_hand': np.random.randn(21, 3),
        'right_hand': np.random.randn(21, 3),
        'face': np.random.randn(468, 3)
    }
    
    features = extractor.extract_features(dummy_data)
    print(f"✅ Enhanced feature extraction test:")
    print(f"   Feature dimension: {features.shape[0]}")
    print(f"   Expected: 54 (18 pose + 24 hands + 12 face)")
    print(f"   Feature range: [{features.min():.3f}, {features.max():.3f}]")
