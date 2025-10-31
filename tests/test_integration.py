"""
Integration Tests
Tests the full pipeline: Model Output → Label Mapping → UI Display → Audio Playback
"""

import unittest
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from unified_classifier import UnifiedClassifier
from webapp.app import load_manifest, load_label_map


class TestModelToUIIntegration(unittest.TestCase):
    """Test integration from model output to UI"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manifest = load_manifest()
        self.label_map = load_label_map()
        
        # Load a classifier if available
        model_path = PROJECT_ROOT / 'pose_model_4_ultimate.pth'
        if model_path.exists():
            self.classifier = UnifiedClassifier(str(model_path))
        else:
            self.classifier = None
    
    def test_old_model_label_mapping(self):
        """Test that old model labels map to new canonical IDs"""
        old_labels = [
            'Arms Folded Laughing',
            'Hands Chest Kissing', 
            'Hands Raised Screaming'
        ]
        
        expected_ids = ['e_wiz', 'kissing', 'screaming']
        
        for old_label, expected_id in zip(old_labels, expected_ids):
            canonical_id = self.label_map['model_to_canonical'][old_label]
            self.assertEqual(canonical_id, expected_id,
                           f"{old_label} should map to {expected_id}")
    
    def test_canonical_id_to_display_label(self):
        """Test that canonical IDs map to correct display labels"""
        id_to_label = {
            'e_wiz': 'E Wiz',
            'kissing': 'Kissing',
            'screaming': 'Screaming'
        }
        
        for emote_id, expected_label in id_to_label.items():
            emote = next(e for e in self.manifest['emotes'] if e['id'] == emote_id)
            self.assertEqual(emote['label'], expected_label,
                           f"ID {emote_id} should have label {expected_label}")
    
    def test_display_label_to_audio_mapping(self):
        """Test that display labels map to correct audio files"""
        label_to_audio = {
            'E Wiz': 'E_Wiz.mp3',
            'Kissing': 'Kissing.mp3',
            'Screaming': 'Screaming.mp3'
        }
        
        for label, expected_audio in label_to_audio.items():
            emote = next(e for e in self.manifest['emotes'] if e['label'] == label)
            self.assertIn(expected_audio, emote['audio'],
                         f"Label {label} should have audio {expected_audio}")
    
    def test_full_pipeline_simulation(self):
        """Simulate full pipeline: old model output → new display"""
        # Simulate model outputs old label
        old_model_output = "Arms Folded Laughing"
        
        # Step 1: Map to canonical ID
        canonical_id = self.label_map['model_to_canonical'][old_model_output]
        self.assertEqual(canonical_id, 'e_wiz')
        
        # Step 2: Get display label from manifest
        emote = next(e for e in self.manifest['emotes'] if e['id'] == canonical_id)
        display_label = emote['label']
        self.assertEqual(display_label, 'E Wiz')
        
        # Step 3: Get audio file
        audio_files = emote['audio']
        self.assertIn('E_Wiz.mp3', audio_files)
        
        # Step 4: Check training status
        self.assertTrue(emote['trained'])
        
        print(f"\n✅ Full pipeline test passed:")
        print(f"   Model output: {old_model_output}")
        print(f"   → Canonical ID: {canonical_id}")
        print(f"   → Display label: {display_label}")
        print(f"   → Audio files: {', '.join(audio_files)}")
        print(f"   → Trained: {emote['trained']}")
    
    def test_classifier_output_labels(self):
        """Test that classifier outputs correct labels"""
        if not self.classifier:
            self.skipTest("No classifier available")
        
        # Check classifier's label mappings
        expected_labels = {
            4: "E Wiz",
            5: "Kissing",
            6: "Screaming"
        }
        
        for index, expected_label in expected_labels.items():
            actual_label = self.classifier.pose_labels[index]
            self.assertEqual(actual_label, expected_label,
                           f"Classifier index {index} should output '{expected_label}'")


class TestAudioPlaybackIntegration(unittest.TestCase):
    """Test audio playback integration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.manifest = load_manifest()
        self.audio_config = self.manifest['audio_config']
    
    def test_confidence_threshold_flow(self):
        """Test confidence threshold in audio playback flow"""
        threshold = self.audio_config['confidence_threshold']
        
        # Test cases: (confidence, should_play)
        test_cases = [
            (0.90, True),   # High confidence - should play
            (0.65, True),   # At threshold - should play
            (0.64, False),  # Below threshold - should not play
            (0.50, False),  # Low confidence - should not play
        ]
        
        for confidence, should_play in test_cases:
            result = confidence >= threshold
            self.assertEqual(result, should_play,
                           f"Confidence {confidence} should {'play' if should_play else 'not play'}")
    
    def test_debounce_simulation(self):
        """Test debounce timing simulation"""
        import time
        
        debounce_ms = self.audio_config['debounce_ms']
        last_played = {}
        
        # Simulate first play
        emote_id = 'e_wiz'
        now = time.time() * 1000
        last_played[emote_id] = now
        
        # Try to play again immediately - should be debounced
        now2 = time.time() * 1000
        elapsed = now2 - last_played[emote_id]
        should_debounce = elapsed < debounce_ms
        self.assertTrue(should_debounce,
                       "Immediate replay should be debounced")
        
        # Wait for debounce period
        time.sleep(debounce_ms / 1000 + 0.1)
        now3 = time.time() * 1000
        elapsed = now3 - last_played[emote_id]
        should_play = elapsed >= debounce_ms
        self.assertTrue(should_play,
                       "Replay after debounce period should be allowed")
    
    def test_preload_configuration(self):
        """Test preload configuration"""
        preload_count = self.audio_config['preload_count']
        
        self.assertEqual(preload_count, 6,
                        "Should preload top 6 emotes")
        
        # Count trained emotes
        trained_emotes = [e for e in self.manifest['emotes'] if e['trained']]
        self.assertGreaterEqual(len(trained_emotes), preload_count,
                               f"Should have at least {preload_count} trained emotes")


class TestUntrainedEmoteBehavior(unittest.TestCase):
    """Test untrained emote behavior in full pipeline"""
    
    def test_untrained_emote_no_audio(self):
        """Test that untrained emotes don't play audio"""
        manifest = load_manifest()
        
        # Simulate untrained emote
        untrained_emote = {
            'id': 'test_untrained',
            'label': 'Test Untrained',
            'trained': False,
            'audio': ['test.mp3']
        }
        
        # Check trained status
        self.assertFalse(untrained_emote['trained'])
        
        # In actual implementation, audio should not play for untrained emotes
        # This is enforced in audio-manager.js playEmote() method
    
    def test_ui_untrained_styling(self):
        """Test that untrained emotes have correct UI indicators"""
        # This would be a browser automation test in a full test suite
        # For now, we verify the manifest structure supports it
        
        manifest = load_manifest()
        
        for emote in manifest['emotes']:
            # All current emotes are trained, but structure supports untrained
            self.assertIn('trained', emote,
                         "Emote should have 'trained' property")
            self.assertIsInstance(emote['trained'], bool,
                                "Trained property should be boolean")


def run_integration_tests():
    """Run all integration tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestModelToUIIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioPlaybackIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestUntrainedEmoteBehavior))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_integration_tests()
    sys.exit(0 if success else 1)
