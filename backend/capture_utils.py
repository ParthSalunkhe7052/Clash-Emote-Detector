"""
Capture Utilities - v2.2
Threading, quality analysis, and helper functions for intelligent data capture

Author: Windsurf Build Agent
Date: October 29, 2025
"""

import cv2
import numpy as np
import threading
import queue
import time
import os
import json
from pathlib import Path
from datetime import datetime


class AsyncFrameWriter:
    """
    Asynchronous frame writer using queue and daemon thread
    Prevents UI lag during high-speed captures
    """
    
    def __init__(self, max_queue_size=100):
        self.queue = queue.Queue(maxsize=max_queue_size)
        self.thread = None
        self.running = False
        self.frames_written = 0
        self.frames_queued = 0
        
    def start(self):
        """Start the writer thread"""
        if self.running:
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._write_loop, daemon=True)
        self.thread.start()
        print("✅ AsyncFrameWriter started")
    
    def stop(self):
        """Stop the writer thread"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5.0)
        print(f"⏹️ AsyncFrameWriter stopped ({self.frames_written} frames written)")
    
    def add_frame(self, frame, filepath, quality=95):
        """
        Add frame to write queue
        
        Args:
            frame: numpy array (image)
            filepath: Path or str where to save
            quality: JPEG quality (1-100)
        """
        try:
            self.queue.put((frame, str(filepath), quality), block=False)
            self.frames_queued += 1
        except queue.Full:
            print("⚠️ Frame queue full, dropping frame")
    
    def _write_loop(self):
        """Background thread loop for writing frames"""
        while self.running or not self.queue.empty():
            try:
                frame, filepath, quality = self.queue.get(timeout=0.5)
                cv2.imwrite(filepath, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
                self.frames_written += 1
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                print(f"❌ Error writing frame: {e}")
    
    def get_queue_size(self):
        """Get current queue size"""
        return self.queue.qsize()
    
    def is_busy(self):
        """Check if queue has pending writes"""
        return not self.queue.empty()


class QualityAnalyzer:
    """
    Lightweight quality analyzer for capture frames
    Checks brightness, motion, and basic quality metrics
    """
    
    def __init__(self):
        self.last_frame = None
        self.brightness_history = []
        self.motion_threshold = 0.02  # 2% motion difference
        
    def analyze_frame(self, frame):
        """
        Analyze frame quality
        
        Returns:
            dict with quality metrics
        """
        if frame is None:
            return None
        
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate brightness (average pixel value)
        brightness = np.mean(gray) / 255.0  # Normalize to 0-1
        self.brightness_history.append(brightness)
        if len(self.brightness_history) > 30:
            self.brightness_history.pop(0)
        
        # Calculate motion (if we have previous frame)
        motion = 0.0
        has_motion = False
        if self.last_frame is not None:
            # Calculate frame difference
            diff = cv2.absdiff(self.last_frame, gray)
            motion = np.mean(diff) / 255.0  # Normalize to 0-1
            has_motion = motion > self.motion_threshold
        
        self.last_frame = gray.copy()
        
        # Quality assessment
        quality = {
            'brightness': brightness,
            'avg_brightness': np.mean(self.brightness_history) if self.brightness_history else brightness,
            'motion': motion,
            'has_motion': has_motion,
            'is_dark': brightness < 0.2,  # Less than 20% brightness
            'is_bright': brightness > 0.8,  # More than 80% brightness
            'is_good': 0.2 <= brightness <= 0.8  # Good lighting range
        }
        
        return quality
    
    def reset(self):
        """Reset analyzer state"""
        self.last_frame = None
        self.brightness_history = []


class CameraDetector:
    """
    Auto-detect available cameras
    """
    
    @staticmethod
    def detect_cameras(max_index=5, timeout=2.0):
        """
        Detect all available camera indices
        
        Args:
            max_index: Maximum camera index to check
            timeout: Timeout per camera check (seconds)
        
        Returns:
            list of available camera indices
        """
        available = []
        
        for i in range(max_index + 1):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Try to read a frame to verify it works
                ret, frame = cap.read()
                if ret and frame is not None:
                    available.append(i)
                    print(f"✅ Camera {i} detected")
            cap.release()
        
        return available
    
    @staticmethod
    def get_camera_info(camera_index):
        """
        Get camera information
        
        Returns:
            dict with camera properties
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return None
        
        info = {
            'index': camera_index,
            'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': int(cap.get(cv2.CAP_PROP_FPS))
        }
        
        cap.release()
        return info


class SettingsManager:
    """
    Manage persistent settings
    """
    
    def __init__(self, settings_file="settings.json"):
        self.settings_file = Path(settings_file)
        self.settings = self.load()
    
    def load(self):
        """Load settings from file"""
        if self.settings_file.exists():
            try:
                with open(self.settings_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"⚠️ Failed to load settings: {e}")
        
        # Default settings
        return {
            'camera_index': 1,
            'last_used': None,
            'target_per_emote': 150,
            'jpeg_quality': 95,
            'auto_advance': False
        }
    
    def save(self):
        """Save settings to file"""
        try:
            self.settings['last_used'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(self.settings_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save settings: {e}")
    
    def get(self, key, default=None):
        """Get setting value"""
        return self.settings.get(key, default)
    
    def set(self, key, value):
        """Set setting value"""
        self.settings[key] = value
        self.save()


class SessionManager:
    """
    Manage capture sessions with timestamps
    """
    
    def __init__(self, base_dir="training_data"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_session = None
        self.session_metadata = {}
    
    def create_session(self, camera_index=0):
        """
        Create new capture session
        
        Returns:
            Path to session directory
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_name = f"session_{timestamp}"
        session_dir = self.base_dir / session_name
        session_dir.mkdir(exist_ok=True)
        
        self.current_session = session_dir
        self.session_metadata = {
            'session_name': session_name,
            'timestamp': timestamp,
            'start_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'camera_index': camera_index,
            'emotes_captured': {},
            'total_frames': 0
        }
        
        # Create emote subdirectories
        emotes = ['laughing', 'yawning', 'crying', 'taunting', 
                  'arms_folded_laughing', 'hands_chest_kissing', 'hands_raised_screaming']
        for emote in emotes:
            (session_dir / emote).mkdir(exist_ok=True)
        
        print(f"📁 Created session: {session_name}")
        return session_dir
    
    def save_session_metadata(self):
        """Save session metadata to JSON"""
        if self.current_session:
            metadata_file = self.current_session / "session_metadata.json"
            self.session_metadata['end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(metadata_file, 'w') as f:
                json.dump(self.session_metadata, f, indent=2)
    
    def get_last_session(self):
        """Get most recent session directory"""
        sessions = sorted(self.base_dir.glob("session_*"), reverse=True)
        return sessions[0] if sessions else None
    
    def list_sessions(self):
        """List all sessions"""
        sessions = sorted(self.base_dir.glob("session_*"), reverse=True)
        return [s.name for s in sessions]


class DiskSpaceChecker:
    """
    Check available disk space before capture
    """
    
    @staticmethod
    def check_space(path, required_mb=500):
        """
        Check if enough disk space is available
        
        Args:
            path: Path to check
            required_mb: Required space in MB
        
        Returns:
            tuple (has_space: bool, available_mb: float)
        """
        try:
            import shutil
            stat = shutil.disk_usage(path)
            available_mb = stat.free / (1024 * 1024)
            has_space = available_mb >= required_mb
            return has_space, available_mb
        except Exception as e:
            print(f"⚠️ Failed to check disk space: {e}")
            return True, 0  # Assume space available if check fails


class CaptureLogger:
    """
    Log capture events to file
    """
    
    def __init__(self, log_dir="logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / "capture.log"
    
    def log(self, message, level="INFO"):
        """Log message with timestamp"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            with open(self.log_file, 'a') as f:
                f.write(log_entry)
        except Exception as e:
            print(f"⚠️ Failed to write log: {e}")
    
    def info(self, message):
        """Log info message"""
        self.log(message, "INFO")
    
    def warning(self, message):
        """Log warning message"""
        self.log(message, "WARNING")
    
    def error(self, message):
        """Log error message"""
        self.log(message, "ERROR")


# Test functions
if __name__ == "__main__":
    print("🧪 Testing Capture Utilities...")
    
    # Test camera detection
    print("\n📷 Detecting cameras...")
    cameras = CameraDetector.detect_cameras()
    print(f"Found cameras: {cameras}")
    
    # Test settings manager
    print("\n⚙️ Testing settings...")
    settings = SettingsManager()
    print(f"Current camera: {settings.get('camera_index')}")
    
    # Test session manager
    print("\n📁 Testing sessions...")
    session_mgr = SessionManager()
    sessions = session_mgr.list_sessions()
    print(f"Existing sessions: {len(sessions)}")
    
    # Test disk space
    print("\n💾 Checking disk space...")
    has_space, available = DiskSpaceChecker.check_space(".", required_mb=500)
    print(f"Space available: {has_space} ({available:.1f} MB)")
    
    print("\n✅ All tests complete")
