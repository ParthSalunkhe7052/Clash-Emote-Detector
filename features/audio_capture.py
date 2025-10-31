"""Audio Capture Module

Author: Parth
Real-time microphone capture for voice emote detection.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Try to import audio libraries (will be installed later)
try:
    import sounddevice as sd
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("sounddevice not installed. Install with: pip install sounddevice")


class AudioCapture:
    """Captures audio from microphone for voice emote detection"""
    
    def __init__(self, sample_rate=16000, clip_duration=2.5):
        """
        Initialize audio capture
        
        Args:
            sample_rate: Audio sample rate in Hz (default: 16kHz)
            clip_duration: Duration of audio clips in seconds
        """
        if not AUDIO_AVAILABLE:
            raise ImportError("sounddevice not available. Install with: pip install sounddevice")
        
        self.sample_rate = sample_rate
        self.clip_duration = clip_duration
        self.clip_samples = int(sample_rate * clip_duration)
        self.is_recording = False
        self.audio_buffer = []
        
        logger.info(f"AudioCapture initialized: {sample_rate}Hz, {clip_duration}s clips")
    
    def start_capture(self):
        """Start capturing audio"""
        if not AUDIO_AVAILABLE:
            logger.error("Cannot start capture - sounddevice not available")
            return False
        
        try:
            self.is_recording = True
            self.audio_buffer = []
            logger.info("Audio capture started")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio capture: {e}")
            return False
    
    def get_audio_clip(self, duration=None):
        """
        Record and return an audio clip
        
        Args:
            duration: Duration in seconds (default: use clip_duration)
            
        Returns:
            np.ndarray: Audio data as float32 array
        """
        if not AUDIO_AVAILABLE:
            logger.error("sounddevice not available")
            return None
        
        try:
            dur = duration or self.clip_duration
            logger.info(f"Recording {dur}s audio clip...")
            
            # Record audio
            audio_data = sd.rec(
                int(dur * self.sample_rate),
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32'
            )
            sd.wait()  # Wait until recording is finished
            
            # Flatten to 1D array
            audio_clip = audio_data.flatten()
            
            logger.info(f"Recorded {len(audio_clip)} samples")
            return audio_clip
            
        except Exception as e:
            logger.error(f"Error recording audio: {e}")
            return None
    
    def detect_voice_activity(self, audio_data, threshold=0.02):
        """
        Simple voice activity detection
        
        Args:
            audio_data: Audio samples
            threshold: Amplitude threshold for voice detection
            
        Returns:
            bool: True if voice detected, False otherwise
        """
        if audio_data is None or len(audio_data) == 0:
            return False
        
        # Calculate RMS (Root Mean Square) energy
        rms = np.sqrt(np.mean(audio_data ** 2))
        
        return rms > threshold
    
    def stop_capture(self):
        """Stop capturing audio"""
        self.is_recording = False
        self.audio_buffer = []
        logger.info("Audio capture stopped")
    
    @staticmethod
    def list_devices():
        """List available audio devices"""
        if not AUDIO_AVAILABLE:
            return []
        
        try:
            devices = sd.query_devices()
            logger.info(f"Found {len(devices)} audio devices")
            return devices
        except Exception as e:
            logger.error(f"Error listing devices: {e}")
            return []


# Simple test function
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    if AUDIO_AVAILABLE:
        print("✅ Audio capture module loaded successfully")
        print("\nAvailable devices:")
        devices = AudioCapture.list_devices()
        for i, dev in enumerate(devices):
            print(f"  {i}: {dev['name']}")
        
        # Test capture
        print("\nTesting audio capture (will record 2.5s)...")
        try:
            capture = AudioCapture()
            audio = capture.get_audio_clip()
            if audio is not None:
                has_voice = capture.detect_voice_activity(audio)
                print(f"✅ Recorded {len(audio)} samples")
                print(f"Voice detected: {has_voice}")
        except Exception as e:
            print(f"❌ Test failed: {e}")
    else:
        print("❌ sounddevice not installed")
        print("Install with: pip install sounddevice")
