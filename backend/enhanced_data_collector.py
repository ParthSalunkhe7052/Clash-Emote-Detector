"""
Enhanced Data Collector - 128-D Embeddings with Quality Features
Saves raw pose landmarks for MediaPipe 128-D embeddings
Includes pose quality indicators and real-time feedback

Author: Windsurf Build Agent  
Date: October 30, 2025
"""

import cv2
import numpy as np
import os
import json
import time
from datetime import datetime
from pathlib import Path
from .holistic_detector import HolisticDetector


class EnhancedDataCollector:
    """
    Advanced data collector that saves raw pose landmarks
    for 128-D MediaPipe embedding extraction
    """
    
    def __init__(self, data_dir="pose_data_v2", detector=None):
        """
        Initialize enhanced data collector
        
        Args:
            data_dir: Directory to store captured data
            detector: HolisticDetector instance (optional)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Use provided detector or create new one
        self.detector = detector if detector is not None else HolisticDetector()
        
        # Emote definitions
        self.emotes = {
            0: "crying",
            1: "laughing",
            2: "taunting",
            3: "yawning",
            4: "arms_folded_laughing",
            5: "hands_chest_kissing",
            6: "hands_raised_screaming"
        }
        
        # Collection state
        self.is_capturing = False
        self.current_emote = 0
        self.target_per_emote = 150
        self.samples_collected = {i: 0 for i in range(7)}
        
        # Quality thresholds (very lenient for easy collection)
        self.min_pose_confidence = 0.2  # Minimum confidence to accept sample (very low threshold)
        self.min_visibility = 0.2  # Minimum landmark visibility
        
        # Load existing progress
        self._load_progress()
        
        print(f"✅ EnhancedDataCollector initialized")
        print(f"📁 Data directory: {self.data_dir.absolute()}")
        print(f"📊 Loaded progress: {sum(self.samples_collected.values())} total samples")
    
    def _load_progress(self):
        """Load existing collection progress"""
        # Count existing .npz files for each emote
        for emote_id, emote_name in self.emotes.items():
            count = 0
            for file in self.data_dir.glob(f"{emote_name}_*.npz"):
                count += 1
            self.samples_collected[emote_id] = count
    
    def _save_progress(self):
        """Save collection metadata"""
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'samples_collected': self.samples_collected,
            'total_samples': sum(self.samples_collected.values()),
            'emotes': self.emotes,
            'target_per_emote': self.target_per_emote
        }
        
        metadata_file = self.data_dir / 'collection_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def capture_sample(self, frame, emote_id):
        """
        Capture a single sample with pose landmarks
        
        Args:
            frame: Camera frame (BGR)
            emote_id: Emote ID (0-6)
            
        Returns:
            dict with 'success', 'message', 'quality' keys
        """
        if emote_id not in self.emotes:
            return {'success': False, 'message': 'Invalid emote ID', 'quality': 0.0}
        
        # Check if target reached
        if self.samples_collected[emote_id] >= self.target_per_emote:
            return {
                'success': False,
                'message': f'Target reached ({self.target_per_emote} samples)',
                'quality': 0.0
            }
        
        # Detect pose
        results = self.detector.detect(frame)
        
        # Check if detection failed (timestamp error recovery)
        if results is None:
            return {'success': False, 'message': 'Detection error, retrying...', 'quality': 0.0}
        
        # Check if pose detected
        if results.pose_landmarks is None:
            return {'success': False, 'message': 'No pose detected', 'quality': 0.0}
        
        # Extract landmarks
        landmarks = []
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z, landmark.visibility])
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Calculate quality score
        quality = self._calculate_quality(landmarks)
        
        # Check quality threshold
        if quality < self.min_pose_confidence:
            return {
                'success': False,
                'message': f'Low quality ({quality:.2f})',
                'quality': quality
            }
        
        # Save sample
        emote_name = self.emotes[emote_id]
        sample_id = self.samples_collected[emote_id]
        filename = f"{emote_name}_{sample_id:04d}.npz"
        filepath = self.data_dir / filename
        
        # Save raw landmarks (for 128-D embedding extraction)
        np.savez_compressed(
            filepath,
            pose=landmarks[:, :3],  # x, y, z (without visibility)
            visibility=landmarks[:, 3],  # visibility scores
            label=emote_id,
            quality=quality,
            timestamp=time.time()
        )
        
        # Update count
        self.samples_collected[emote_id] += 1
        self._save_progress()
        
        return {
            'success': True,
            'message': f'Sample {sample_id + 1}/{self.target_per_emote} saved',
            'quality': quality
        }
    
    def _calculate_quality(self, landmarks):
        """
        Calculate pose quality score (0-1)
        Based on CORE BODY landmarks only (not hands/face)
        This ensures emotes without hands still get good quality scores
        """
        # Focus on CORE BODY landmarks only (torso + head)
        # 0=nose, 11=left_shoulder, 12=right_shoulder, 23=left_hip, 24=right_hip
        core_landmarks = [0, 11, 12, 23, 24]
        
        # Get visibility of core landmarks
        visibilities = landmarks[core_landmarks, 3]
        avg_visibility = np.mean(visibilities)
        
        # Check if core landmarks are within frame
        core_positions = landmarks[core_landmarks, :2]  # x, y coordinates
        in_frame = np.all((core_positions >= 0) & (core_positions <= 1))
        
        # Quality score based on core body only
        # This allows emotes without hands to still have good quality
        quality = avg_visibility if in_frame else avg_visibility * 0.7
        
        return float(quality)
    
    def get_progress(self):
        """Get collection progress for all emotes"""
        return {
            'samples_collected': self.samples_collected,
            'target_per_emote': self.target_per_emote,
            'total_samples': sum(self.samples_collected.values()),
            'emotes': self.emotes
        }
    
    def reset_emote(self, emote_id):
        """
        Reset collection for a specific emote
        
        Args:
            emote_id: Emote ID to reset
            
        Returns:
            dict with 'success' and 'message'
        """
        if emote_id not in self.emotes:
            return {'success': False, 'message': 'Invalid emote ID'}
        
        emote_name = self.emotes[emote_id]
        
        # Delete all files for this emote
        deleted_count = 0
        for file in self.data_dir.glob(f"{emote_name}_*.npz"):
            try:
                file.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file}: {e}")
        
        # Reset counter
        self.samples_collected[emote_id] = 0
        self._save_progress()
        
        return {
            'success': True,
            'message': f'Deleted {deleted_count} samples for {emote_name}'
        }
    
    def reset_all(self):
        """
        Reset all collected data
        
        Returns:
            dict with 'success' and 'message'
        """
        total_deleted = 0
        
        # Delete all .npz files
        for file in self.data_dir.glob("*.npz"):
            try:
                file.unlink()
                total_deleted += 1
            except Exception as e:
                print(f"⚠️ Error deleting {file}: {e}")
        
        # Reset all counters
        self.samples_collected = {i: 0 for i in range(7)}
        self._save_progress()
        
        return {
            'success': True,
            'message': f'Deleted {total_deleted} total samples'
        }
    
    def export_training_data(self):
        """
        Export all collected data as training-ready arrays
        
        Returns:
            dict with 'success', 'message', 'landmarks', 'labels'
        """
        all_landmarks = []
        all_labels = []
        
        # Load all .npz files
        for emote_id, emote_name in self.emotes.items():
            for file in sorted(self.data_dir.glob(f"{emote_name}_*.npz")):
                try:
                    data = np.load(file)
                    all_landmarks.append(data['pose'])
                    all_labels.append(emote_id)
                except Exception as e:
                    print(f"⚠️ Error loading {file}: {e}")
        
        if len(all_landmarks) == 0:
            return {
                'success': False,
                'message': 'No data collected yet',
                'landmarks': None,
                'labels': None
            }
        
        landmarks_array = np.array(all_landmarks, dtype=np.float32)
        labels_array = np.array(all_labels, dtype=np.int64)
        
        # Save as .npy files
        np.save(self.data_dir / 'pose_landmarks.npy', landmarks_array)
        np.save(self.data_dir / 'pose_labels.npy', labels_array)
        
        return {
            'success': True,
            'message': f'Exported {len(all_landmarks)} samples',
            'landmarks': landmarks_array,
            'labels': labels_array,
            'num_samples': len(all_landmarks)
        }
    
    def get_quality_stats(self):
        """Get quality statistics for collected data"""
        qualities = []
        
        for file in self.data_dir.glob("*.npz"):
            try:
                data = np.load(file)
                if 'quality' in data:
                    qualities.append(float(data['quality']))
            except:
                pass
        
        if not qualities:
            return {
                'mean': 0.0,
                'min': 0.0,
                'max': 0.0,
                'count': 0
            }
        
        return {
            'mean': float(np.mean(qualities)),
            'min': float(np.min(qualities)),
            'max': float(np.max(qualities)),
            'count': len(qualities)
        }
    
    def __del__(self):
        """Cleanup"""
        # Detector cleanup is handled by the detector itself
        pass
