# Changelog

All notable changes to the Clash Royale Emote Detector project.

## [2.2.0] - 2025-10-30

### Added
- **Model 4 Ultimate**: Advanced 128-D neural network with residual connections and attention
- **Enhanced Data Collection**: Modern web interface for pose data collection
- **Manage Emotes Panel**: Upload and manage custom emotes with images and sounds
- **Settings Page**: Configure camera, model selection, and detection parameters
- **Custom Emotes**: Support for user-defined emotes with manifest system
- **Audio System**: Dynamic audio playback for detected emotes
- **Multiple Model Support**: Switch between RandomForest, Neural Net, Embedding, and Model 4 models

### Changed
- Improved camera management with proper release on page navigation
- Better error handling across all endpoints
- Enhanced UI with modern glassmorphism design
- Optimized pose detection pipeline

### Fixed
- Camera not releasing when switching pages
- Audio playback conflicts
- Model switching without server restart
- localStorage persistence issues

## [2.0.0] - 2025-10-28

### Added
- Flask web interface with real-time Socket.IO communication
- Modern gradient UI with TailwindCSS
- Camera toggle functionality
- Real-time emote detection and display
- Audio playback system

### Changed
- Migrated from standalone Python app to web-based interface
- Improved MediaPipe integration
- Better frame processing with quality control

## [1.0.0] - Initial Release

### Added
- MediaPipe holistic detection (pose, face, hands)
- RandomForest classifier for emote recognition
- 4 base Clash Royale emotes (Laughing, Yawning, Crying, Taunting)
- Data collection tool
- Basic Python window interface
