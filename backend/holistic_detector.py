"""MediaPipe Holistic Detector Module

Author: Parth
Wrapper for MediaPipe Holistic landmark detection.
"""

import cv2
import mediapipe as mp
import numpy as np


class HolisticDetector:
    def __init__(self):
        """Initialize MediaPipe Holistic detector"""
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize holistic model with lower thresholds for easier detection
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.3,  # Lowered from 0.5
            min_tracking_confidence=0.3    # Lowered from 0.5
        )
    
    def detect(self, frame):
        """
        Detect holistic landmarks in a frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            results: MediaPipe holistic results
        """
        try:
            # Create a COPY of the frame to avoid timestamp issues
            frame_copy = frame.copy()
            
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
            
            # Make frame non-writable for better performance
            rgb_frame.flags.writeable = False
            
            # Process the frame
            results = self.holistic.process(rgb_frame)
            
            # Make frame writable again
            rgb_frame.flags.writeable = True
            
            return results
        except Exception as e:
            # If timestamp error, reset the holistic instance and retry
            if "timestamp" in str(e).lower():
                # Reinitialize holistic to clear timestamp state
                self.holistic.close()
                self.holistic = self.mp_holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    refine_face_landmarks=False,
                    min_detection_confidence=0.3,
                    min_tracking_confidence=0.3
                )
                # Return None to indicate retry needed
                return None
            else:
                raise
    
    def draw_landmarks(self, frame, results):
        """
        Draw landmarks on the frame
        
        Args:
            frame: Input frame
            results: MediaPipe holistic results
            
        Returns:
            frame: Frame with landmarks drawn
        """
        # Draw pose landmarks with thicker lines
        thick_pose_connections = self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=4)
        thick_pose_landmarks = self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=6, circle_radius=4)
        
        self.mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            self.mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=thick_pose_landmarks,
            connection_drawing_spec=thick_pose_connections
        )
        
        # Draw minimal face landmarks (just basic points)
        if results.face_landmarks:
            # Only draw a few key facial points for basic face detection
            face_landmarks = results.face_landmarks.landmark
            
            # Draw just a few key points: nose tip and corners of mouth
            key_points = [1, 13, 14, 17, 18]  # Nose tip, mouth corners, eye corners
            
            for point_idx in key_points:
                if point_idx < len(face_landmarks):
                    point = face_landmarks[point_idx]
                    h, w, c = frame.shape
                    x, y = int(point.x * w), int(point.y * h)
                    cv2.circle(frame, (x, y), 3, (255, 255, 255), -1)
        
        # Draw hand landmarks with thicker lines
        thick_hand_connections = self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=3)
        thick_hand_landmarks = self.mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=4, circle_radius=3)
        
        # Draw left hand landmarks with thicker lines
        self.mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            thick_hand_landmarks,
            thick_hand_connections
        )
        
        # Draw right hand landmarks with thicker lines
        self.mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            self.mp_holistic.HAND_CONNECTIONS,
            thick_hand_landmarks,
            thick_hand_connections
        )
        
        return frame
    
    def get_landmark_data(self, results):
        """
        Extract landmark coordinates from results
        
        Args:
            results: MediaPipe holistic results
            
        Returns:
            dict: Dictionary containing landmark data
        """
        landmark_data = {
            'pose': None,
            'face': None,
            'left_hand': None,
            'right_hand': None
        }
        
        if results.pose_landmarks:
            # Include visibility as 4th column for Model 4 compatibility
            landmark_data['pose'] = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark])
        
        if results.face_landmarks:
            landmark_data['face'] = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark])
        
        if results.left_hand_landmarks:
            landmark_data['left_hand'] = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark])
        
        if results.right_hand_landmarks:
            landmark_data['right_hand'] = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark])
        
        return landmark_data
    
    def release(self):
        """Release resources"""
        if hasattr(self, 'holistic'):
            self.holistic.close()
