"""
Unit Tests for Audio Manager Backend API
Tests manifest loading, emote management, and audio file validation
"""

import unittest
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from webapp.app import app, load_manifest, save_manifest, load_label_map


class TestAudioManagerAPI(unittest.TestCase):
    """Test audio manager backend APIs"""
    
    def setUp(self):
        """Set up test client"""
        self.app = app
        self.app.config['TESTING'] = True
        self.client = self.app.test_client()
    
    def test_load_manifest(self):
        """Test manifest loading"""
        manifest = load_manifest()
        
        self.assertIsNotNone(manifest, "Manifest should load successfully")
        self.assertIn('version', manifest, "Manifest should have version")
        self.assertIn('emotes', manifest, "Manifest should have emotes array")
        self.assertIn('audio_config', manifest, "Manifest should have audio_config")
        
        # Check audio config defaults
        config = manifest['audio_config']
        self.assertEqual(config['confidence_threshold'], 0.65)
        self.assertEqual(config['debounce_ms'], 1500)
        self.assertEqual(config['preload_count'], 6)
        
    def test_emotes_list_endpoint(self):
        """Test /api/emotes/list endpoint"""
        response = self.client.get('/api/emotes/list')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'], "API should return success")
        self.assertIn('emotes', data, "Response should include emotes")
        
        # Check emote structure
        if len(data['emotes']) > 0:
            emote = data['emotes'][0]
            self.assertIn('id', emote)
            self.assertIn('label', emote)
            self.assertIn('trained', emote)
            self.assertIn('audio', emote)
            self.assertIn('filename', emote)
    
    def test_manifest_endpoint(self):
        """Test /api/emotes/manifest endpoint"""
        response = self.client.get('/api/emotes/manifest')
        data = json.loads(response.data)
        
        self.assertTrue(data['success'], "API should return success")
        self.assertIn('manifest', data, "Response should include manifest")
        
        manifest = data['manifest']
        self.assertIn('emotes', manifest)
        self.assertIn('audio_config', manifest)
    
    def test_label_mapping(self):
        """Test model label mapping"""
        label_map = load_label_map()
        
        self.assertIsNotNone(label_map, "Label map should load")
        self.assertIn('model_to_canonical', label_map)
        
        mapping = label_map['model_to_canonical']
        
        # Test old labels map to new IDs
        self.assertEqual(mapping['Arms Folded Laughing'], 'e_wiz')
        self.assertEqual(mapping['Hands Chest Kissing'], 'kissing')
        self.assertEqual(mapping['Hands Raised Screaming'], 'screaming')
        
        # Test new labels also map correctly
        self.assertEqual(mapping['E Wiz'], 'e_wiz')
        self.assertEqual(mapping['Kissing'], 'kissing')
        self.assertEqual(mapping['Screaming'], 'screaming')
    
    def test_emote_label_renaming(self):
        """Test that emote labels have been renamed correctly"""
        manifest = load_manifest()
        
        emote_labels = [e['label'] for e in manifest['emotes']]
        
        # Check new labels exist
        self.assertIn('E Wiz', emote_labels)
        self.assertIn('Kissing', emote_labels)
        self.assertIn('Screaming', emote_labels)
        
        # Check old labels don't exist
        self.assertNotIn('Arms Folded Laughing', emote_labels)
        self.assertNotIn('Hands Chest Kissing', emote_labels)
        self.assertNotIn('Hands Raised Screaming', emote_labels)
    
    def test_audio_file_mappings(self):
        """Test that audio file mappings are correct"""
        manifest = load_manifest()
        
        # Find renamed emotes
        e_wiz = next(e for e in manifest['emotes'] if e['id'] == 'e_wiz')
        kissing = next(e for e in manifest['emotes'] if e['id'] == 'kissing')
        screaming = next(e for e in manifest['emotes'] if e['id'] == 'screaming')
        
        # Check audio file names match new format
        self.assertIn('E_Wiz.mp3', e_wiz['audio'])
        self.assertIn('Kissing.mp3', kissing['audio'])
        self.assertIn('Screaming.mp3', screaming['audio'])
    
    def test_trained_status(self):
        """Test that all current emotes are marked as trained"""
        manifest = load_manifest()
        
        for emote in manifest['emotes']:
            # All 7 emotes should be trained by default
            self.assertTrue(emote['trained'], 
                          f"Emote {emote['label']} should be trained by default")
    
    def test_update_training_status(self):
        """Test updating training status endpoint"""
        # Try to update an emote's training status
        response = self.client.post('/api/emotes/update_training_status',
                                   json={'emote_id': 'laughing', 'trained': False})
        data = json.loads(response.data)
        
        self.assertTrue(data['success'], "Update should succeed")
        
        # Verify it was updated
        manifest = load_manifest()
        laughing = next(e for e in manifest['emotes'] if e['id'] == 'laughing')
        self.assertFalse(laughing['trained'], "Training status should be updated")
        
        # Reset it back
        response = self.client.post('/api/emotes/update_training_status',
                                   json={'emote_id': 'laughing', 'trained': True})
        data = json.loads(response.data)
        self.assertTrue(data['success'], "Reset should succeed")


class TestAudioFilesExist(unittest.TestCase):
    """Test that audio files exist for trained emotes"""
    
    def setUp(self):
        """Load manifest"""
        self.manifest = load_manifest()
        self.sounds_dir = PROJECT_ROOT / 'sounds'
    
    def test_trained_emotes_have_audio(self):
        """Test that all trained emotes have at least one audio file"""
        missing = []
        
        for emote in self.manifest['emotes']:
            if emote['trained']:
                # Check if at least one audio file exists
                found = False
                for audio_file in emote['audio']:
                    audio_path = self.sounds_dir / audio_file
                    if audio_path.exists():
                        found = True
                        break
                
                if not found:
                    missing.append({
                        'emote': emote['label'],
                        'expected': emote['audio']
                    })
        
        self.assertEqual(len(missing), 0, 
                        f"Missing audio files: {missing}")
    
    def test_default_missing_audio_exists(self):
        """Test that default missing audio file exists"""
        default_audio = self.manifest['audio_config']['default_missing_audio']
        # Remove leading slash if present
        default_audio = default_audio.lstrip('/')
        default_path = PROJECT_ROOT / default_audio
        
        self.assertTrue(default_path.exists(),
                       f"Default missing audio should exist at {default_path}")


class TestDebounceLogic(unittest.TestCase):
    """Test debounce logic for audio playback"""
    
    def test_debounce_timing(self):
        """Test that debounce timing is correct"""
        manifest = load_manifest()
        debounce_ms = manifest['audio_config']['debounce_ms']
        
        self.assertEqual(debounce_ms, 1500, 
                        "Default debounce should be 1500ms")
        self.assertIsInstance(debounce_ms, int, 
                             "Debounce should be an integer")
    
    def test_confidence_threshold(self):
        """Test confidence threshold configuration"""
        manifest = load_manifest()
        threshold = manifest['audio_config']['confidence_threshold']
        
        self.assertEqual(threshold, 0.65, 
                        "Default confidence threshold should be 0.65")
        self.assertGreaterEqual(threshold, 0.0, 
                               "Threshold should be >= 0")
        self.assertLessEqual(threshold, 1.0, 
                            "Threshold should be <= 1")


class TestUntrainedEmotesBehavior(unittest.TestCase):
    """Test untrained emotes behavior"""
    
    def test_untrained_emotes_disabled(self):
        """Test that untrained emotes are properly marked"""
        # Temporarily mark an emote as untrained
        manifest = load_manifest()
        
        # Find an emote and mark it untrained
        test_emote = manifest['emotes'][0]
        original_status = test_emote['trained']
        test_emote['trained'] = False
        
        # Save and reload
        save_manifest(manifest)
        reloaded = load_manifest()
        
        # Check it's marked as untrained
        reloaded_emote = next(e for e in reloaded['emotes'] 
                             if e['id'] == test_emote['id'])
        self.assertFalse(reloaded_emote['trained'], 
                        "Emote should be marked as untrained")
        
        # Restore original status
        test_emote['trained'] = original_status
        save_manifest(manifest)


def run_tests():
    """Run all tests"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAudioManagerAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestAudioFilesExist))
    suite.addTests(loader.loadTestsFromTestCase(TestDebounceLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestUntrainedEmotesBehavior))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
