"""Clash Royale Emote Detector - Web Application

Author: Parth
Flask web server for real-time emote detection.
"""

from flask import Flask, render_template, Response, jsonify, send_from_directory, request
from flask_socketio import SocketIO, emit
import cv2
import sys
import os
import json
import time
import logging
import warnings

# Suppress protobuf deprecation warnings
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.holistic_detector import HolisticDetector
from backend.unified_classifier import UnifiedClassifier
from backend.enhanced_data_collector import EnhancedDataCollector
from backend.enhanced_visual_features import EnhancedVisualFeatureExtractor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'clash-emote-detector-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Get parent directory (project root)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Model management
import threading
import numpy as np
model_lock = threading.Lock()  # Thread-safe model switching
current_model_name = 'backend/models/pose_model_4_ultimate.pth'  # Default model

# Initialize detector and unified classifier
try:
    detector = HolisticDetector()
    classifier = UnifiedClassifier(model_path=os.path.join(PROJECT_ROOT, current_model_name))
    logger.info(f"✅ Detector and unified classifier initialized successfully")
    logger.info(f"📦 Active model: {current_model_name} ({classifier.model_type})")
except Exception as e:
    logger.error(f"❌ Failed to initialize: {e}")
    detector = None
    classifier = None

# Initialize enhanced data collector
try:
    data_collector = EnhancedDataCollector()
    feature_extractor = EnhancedVisualFeatureExtractor()
    logger.info("✅ Enhanced data collector initialized")
except Exception as e:
    logger.error(f"❌ Failed to initialize data collector: {e}")
    data_collector = None
    feature_extractor = None

# Initialize enhanced data collector for capture page (uses shared detector)
try:
    from backend.enhanced_data_collector import EnhancedDataCollector
    enhanced_collector = EnhancedDataCollector(data_dir="pose_data_v2", detector=detector)
    logger.info("✅ Enhanced data collector initialized (128-D embeddings)")
except Exception as e:
    logger.error(f"❌ Failed to initialize enhanced collector: {e}")
    enhanced_collector = None

# Simple collector removed - using enhanced collector only
simple_collector = None

# ========== NEW CAMERA SYSTEM ==========
# Completely rewritten camera management with proper Windows support

# Camera state
camera = None
camera_active = True
current_camera_index = 0
camera_lock = threading.Lock()

# Temporal smoothing for stable predictions
from collections import deque
prediction_buffer = deque(maxlen=10)
confidence_buffer = deque(maxlen=10)


def detect_available_cameras():
    """
    Return static list of cameras without actually opening them.
    This prevents camera locking issues.
    
    We assume:
    - Index 0: Laptop Camera (always present)
    - Index 1: DroidCam (may or may not be present)
    
    The actual camera opening happens in get_camera() which handles errors properly.
    """
    logger.info("🔍 Listing available cameras...")
    
    # Return static list - don't actually open cameras here
    # This prevents the camera lock issue
    available = [
        {
            'index': 0,
            'name': 'Laptop Camera',
            'resolution': '640x480'
        },
        {
            'index': 1,
            'name': 'DroidCam',
            'resolution': '640x480'
        }
    ]
    
    logger.info(f"📷 Available: Laptop Camera (0), DroidCam (1)")
    return available


def get_camera():
    """
    Get or initialize camera with appropriate backend
    Tries multiple backends if first one fails
    Returns existing camera if already open, otherwise creates new one
    """
    global camera, current_camera_index
    
    with camera_lock:
        # If camera exists, test if it's actually working
        if camera is not None and camera.isOpened():
            # Quick test read
            try:
                ret, test_frame = camera.read()
                if ret and test_frame is not None:
                    return camera
                else:
                    # Camera is open but not working, release it
                    logger.warning("Camera was open but not reading frames, releasing...")
                    camera.release()
                    camera = None
            except Exception as e:
                logger.warning(f"Camera test failed: {e}, releasing...")
                try:
                    camera.release()
                except:
                    pass
                camera = None
        
        # Need to open camera
        logger.info(f"📷 Opening camera {current_camera_index}...")
        
        # Try multiple backends in order of preference
        backends_to_try = []
        if current_camera_index == 0:
            # Laptop camera: try DirectShow, then Media Foundation, then Auto
            backends_to_try = [
                (cv2.CAP_DSHOW, "DirectShow"),
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Auto")
            ]
        else:
            # DroidCam: try Media Foundation, then Auto
            backends_to_try = [
                (cv2.CAP_MSMF, "Media Foundation"),
                (cv2.CAP_ANY, "Auto")
            ]
        
        # Try each backend until one works
        for backend, backend_name in backends_to_try:
            try:
                logger.info(f"   Trying {backend_name} backend...")
                camera = cv2.VideoCapture(current_camera_index, backend)
                
                if not camera.isOpened():
                    logger.warning(f"   ⚠️ {backend_name} failed to open camera")
                    camera = None
                    continue
                
                # CRITICAL: Give camera time to initialize before reading
                time.sleep(0.5)
                
                # Configure camera settings BEFORE first read
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                camera.set(cv2.CAP_PROP_FPS, 30)
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                
                # Give settings time to apply
                time.sleep(0.2)
                
                # Clear any buffered frames
                for _ in range(5):
                    camera.grab()
                
                # Now try to read frames
                success_count = 0
                for i in range(10):
                    ret, test_frame = camera.read()
                    if ret and test_frame is not None:
                        success_count += 1
                        if success_count >= 3:
                            break
                    time.sleep(0.1)
                
                if success_count < 3:
                    logger.warning(f"   ⚠️ {backend_name} opened but cannot read frames reliably")
                    camera.release()
                    camera = None
                    continue
                
                # SUCCESS!
                logger.info(f"✅ Camera {current_camera_index} ready ({backend_name})")
                return camera
                
            except Exception as e:
                logger.warning(f"   ⚠️ {backend_name} exception: {e}")
                if camera is not None:
                    try:
                        camera.release()
                    except:
                        pass
                camera = None
                continue
        
        # All backends failed
        logger.error(f"❌ All backends failed for camera {current_camera_index}")
        return None


def release_camera():
    """Release camera resources safely"""
    global camera
    
    with camera_lock:
        if camera is not None:
            try:
                camera.release()
                logger.info("📷 Camera released")
            except Exception as e:
                logger.warning(f"⚠️ Error releasing camera: {e}")
            finally:
                camera = None


def switch_camera(camera_index):
    """
    Switch to a different camera - uses get_camera() logic
    """
    global camera, current_camera_index
    
    logger.info(f"🔄 Switching from camera {current_camera_index} to {camera_index}")
    
    with camera_lock:
        # Release current camera
        if camera is not None:
            try:
                camera.release()
            except Exception as e:
                logger.warning(f"⚠️ Error releasing old camera: {e}")
            camera = None
        
        # Wait for camera to fully release
        time.sleep(0.5)
        
        # Update index
        current_camera_index = camera_index
    
    # Use get_camera() which has all the multi-backend logic
    new_camera = get_camera()
    
    if new_camera is not None:
        logger.info(f"✅ Successfully switched to camera {camera_index}")
        return True
    else:
        logger.error(f"❌ Failed to switch to camera {camera_index}")
        return False


@app.route('/')
def index():
    """Camera detector page (main page)"""
    return render_template('index.html')


@app.route('/voice')
def voice():
    """Voice guesser page"""
    return render_template('voice.html')


@app.route('/emotes')
def emotes():
    """Custom emotes manager"""
    return render_template('emotes.html')


@app.route('/settings')
def settings():
    """Settings page"""
    return render_template('settings.html')


@app.route('/collect')
def collect():
    """Advanced data collection page (old)"""
    return render_template('collect.html')


@app.route('/capture')
def capture():
    """Enhanced data capture page with 128-D embeddings"""
    return render_template('capture_enhanced.html')

@app.route('/capture_old')
def capture_old():
    """Old simple data capture page (legacy)"""
    return render_template('capture.html')


@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve emote images from project root images folder"""
    return send_from_directory(os.path.join(PROJECT_ROOT, 'images'), filename)


@app.route('/sounds/<path:filename>')
def serve_sound(filename):
    """Serve emote sounds from project root sounds folder"""
    return send_from_directory(os.path.join(PROJECT_ROOT, 'sounds'), filename)


@app.route('/favicon.ico')
def favicon():
    """Return empty response for favicon to prevent 404 errors"""
    return '', 204


@app.route('/.well-known/<path:path>')
def well_known(path):
    """Return empty response for Chrome DevTools requests"""
    return '', 204


def create_placeholder_frame(message="No Camera Input"):
    """Create a placeholder frame when camera is not available"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background
    
    # Add text
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(message, font, 1, 2)[0]
    text_x = (640 - text_size[0]) // 2
    text_y = (480 + text_size[1]) // 2
    cv2.putText(frame, message, (text_x, text_y), font, 1, (255, 255, 255), 2)
    
    # Add icon
    cv2.circle(frame, (320, 200), 50, (100, 100, 100), -1)
    cv2.line(frame, (290, 180), (350, 220), (200, 200, 200), 3)
    cv2.line(frame, (290, 220), (350, 180), (200, 200, 200), 3)
    
    return frame


def generate_frames():
    """Generate video frames for camera detection with improved error handling"""
    global camera_active

    if detector is None or classifier is None:
        logger.error("Detector or classifier not initialized")
        placeholder = create_placeholder_frame("Detector Not Initialized")
        ret, buffer = cv2.imencode('.jpg', placeholder)
        if ret:
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        return

    # Get camera once at start
    cap = get_camera()
    if cap is None:
        logger.error("Failed to initialize camera")
        placeholder = create_placeholder_frame("Camera Not Available")
        ret, buffer = cv2.imencode('.jpg', placeholder)
        if ret:
            frame_bytes = buffer.tobytes()
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
        return

    last_emote = None
    consecutive_failures = 0
    max_failures = 50  # Increased from 30 to 50 to handle longer initialization
    warmup_frames = 10  # Skip first few frames to allow camera to stabilize
    frame_count = 0

    try:
        while True:
            # Check if camera should be active
            if not camera_active:
                socketio.emit('emote_detected', {
                    'emote': "No Pose",
                    'confidence': 0.0
                })
                placeholder = create_placeholder_frame("Camera OFF")
                ret, buffer = cv2.imencode('.jpg', placeholder)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(0.1)
                continue

            # Read frame from camera
            success, frame = cap.read()
            if not success or frame is None:
                consecutive_failures += 1
                
                # Only log every 10th failure to avoid spam
                if consecutive_failures % 10 == 0:
                    logger.warning(f"Failed to read frame (attempt {consecutive_failures}/{max_failures})")
                
                if consecutive_failures >= max_failures:
                    logger.error("Too many failures, camera disconnected")
                    placeholder = create_placeholder_frame("Camera Connection Lost")
                    ret, buffer = cv2.imencode('.jpg', placeholder)
                    if ret:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    break
                
                time.sleep(0.15)  # Increased to give camera more time to recover
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            frame_count += 1
            
            # Skip warm-up frames
            if frame_count <= warmup_frames:
                time.sleep(0.033)  # ~30 FPS
                continue

            # Mirror frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Process frame with detector
            try:
                results = detector.detect(frame)
                frame = detector.draw_landmarks(frame, results)

                landmark_data = detector.get_landmark_data(results)
                pose_landmarks = landmark_data.get('pose')

                pose_prediction = "No Pose"
                confidence = 0.0

                if pose_landmarks is not None:
                    # Pass raw landmarks directly - classifier will handle feature extraction
                    pose_prediction, confidence = classifier.predict(pose_landmarks, landmark_data=landmark_data)
                    
                    # Data collection integration
                    if data_collector and feature_extractor:
                        try:
                            quality = data_collector.assess_quality(landmark_data, frame)
                            
                            quality_json = {}
                            for k, v in quality.items():
                                if isinstance(v, (bool, np.bool_)):
                                    quality_json[k] = bool(v)
                                elif isinstance(v, (int, np.integer)):
                                    quality_json[k] = int(v)
                                elif isinstance(v, (float, np.floating)):
                                    quality_json[k] = float(v)
                                else:
                                    quality_json[k] = str(v)
                            socketio.emit('quality_feedback', quality_json)
                        except Exception as qe:
                            logger.debug(f"Quality feedback error: {qe}")
                        
                        # Data collection disabled on home page - only for /capture page
                        pass

                    # Temporal smoothing
                    prediction_buffer.append(pose_prediction)
                    confidence_buffer.append(confidence)

                    if len(prediction_buffer) >= 3:
                        from collections import Counter
                        vote_counts = Counter(prediction_buffer)
                        smoothed_prediction = vote_counts.most_common(1)[0][0]
                        smoothed_confidence = np.mean([c for p, c in zip(prediction_buffer, confidence_buffer) if p == smoothed_prediction])
                    else:
                        smoothed_prediction = pose_prediction
                        smoothed_confidence = confidence

                    # Draw prediction on frame
                    if smoothed_confidence > 0.65:
                        color = (0, 255, 0)
                    elif smoothed_confidence > 0.35:
                        color = (0, 255, 255)
                    else:
                        color = (0, 0, 255)

                    cv2.putText(frame, f"{smoothed_prediction} ({smoothed_confidence:.2f})",
                               (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                    if smoothed_prediction != last_emote and smoothed_confidence > 0.35:
                        socketio.emit('emote_detected', {
                            'emote': smoothed_prediction,
                            'confidence': float(smoothed_confidence)
                        })
                        last_emote = smoothed_prediction
                else:
                    if last_emote != "No Pose":
                        socketio.emit('emote_detected', {
                            'emote': "No Pose",
                            'confidence': 0.0
                        })
                        last_emote = "No Pose"

            except Exception as e:
                logger.error(f"Error processing frame: {e}", exc_info=False)

            # Encode and yield frame
            try:
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Exception as e:
                logger.error(f"Error encoding frame: {e}")
                continue

    except Exception as e:
        logger.error(f"Critical error in generate_frames: {e}")
    finally:
        logger.info("Frame generation stopped")


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print('Client connected')
    emit('status', {'message': 'Connected to server'})


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print('Client disconnected')


@socketio.on('toggle_camera')
def handle_camera_toggle(data):
    """Handle camera on/off toggle - actually releases camera when off"""
    global camera_active, camera
    
    new_state = data.get('active', True)
    
    if not new_state and camera_active:
        # Turning OFF - release the camera
        logger.info("🔴 Turning camera OFF - releasing hardware")
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
        camera_active = False
        emit('camera_status', {'active': False})
        
    elif new_state and not camera_active:
        # Turning ON - camera will be reinitialized on next frame request
        logger.info("🟢 Turning camera ON")
        camera_active = True
        emit('camera_status', {'active': True})
    
    logger.info(f"Camera status: {'ON' if camera_active else 'OFF'}")


# ========== Camera Management API ==========

@app.route('/api/list_cameras')
def list_cameras():
    """List all available cameras"""
    try:
        cameras = detect_available_cameras()
        return jsonify({
            'success': True,
            'cameras': cameras,
            'current_index': current_camera_index
        })
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/switch_camera', methods=['POST'])
def api_switch_camera():
    """Switch to a different camera"""
    try:
        from flask import request
        data = request.json
        camera_index = data.get('camera_index', 0)
        
        success = switch_camera(camera_index)
        
        return jsonify({
            'success': success,
            'camera_index': current_camera_index,
            'message': f'Switched to camera {current_camera_index}' if success else 'Failed to switch camera'
        })
    except Exception as e:
        logger.error(f"Error switching camera: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/camera/force_release', methods=['POST'])
def force_release_camera():
    """Force release camera - useful when camera is stuck"""
    try:
        global camera, camera_active
        
        logger.info("🔧 Force releasing camera...")
        
        # First, ensure camera is set to active state
        camera_active = True
        logger.info("   Setting camera_active = True")
        
        with camera_lock:
            if camera is not None:
                try:
                    camera.release()
                    logger.info("   Released camera object")
                except Exception as e:
                    logger.warning(f"   Error releasing: {e}")
                
                camera = None
        
        # Give OS time to release resources (increased from 1.0 to 2.0 seconds)
        time.sleep(2.0)
        
        # Try to reinitialize
        logger.info("   Reinitializing camera...")
        new_camera = get_camera()
        
        if new_camera is not None:
            # Test that camera can actually grab frames
            test_success = False
            for i in range(10):
                ret, frame = new_camera.read()
                if ret and frame is not None:
                    test_success = True
                    break
                time.sleep(0.1)
            
            if test_success:
                logger.info("✅ Camera successfully reset and reinitialized")
                return jsonify({
                    'success': True,
                    'message': 'Camera reset successful'
                })
            else:
                logger.error("❌ Camera opened but cannot read frames")
                return jsonify({
                    'success': False,
                    'message': 'Camera opened but cannot read frames'
                })
        else:
            logger.error("❌ Failed to reinitialize camera after reset")
            return jsonify({
                'success': False,
                'message': 'Camera reset but failed to reinitialize'
            })
            
    except Exception as e:
        logger.error(f"Error in force release: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# ========== Model Selection Routes (v2.1) ==========

@app.route('/api/models')
def list_models():
    """List all available model files (.pth and .pkl)"""
    try:
        models = []
        # Search for model files in backend/models directory
        models_dir = os.path.join(PROJECT_ROOT, 'backend', 'models')
        for file in os.listdir(models_dir):
            if file.endswith('.pth') or file.endswith('.pkl'):
                file_path = os.path.join(models_dir, file)
                file_size = os.path.getsize(file_path) / 1024  # KB

                # Determine model type
                if file.endswith('.pkl'):
                    model_type = 'RandomForest'
                elif file.endswith('.pth'):
                    # Check model type by filename
                    if 'model_4' in file.lower() or 'ultimate' in file.lower():
                        model_type = '🚀 Model 4 Ultimate (128-D)'
                    elif 'embedding' in file.lower():
                        model_type = 'Embedding (128-D)'
                    elif 'enhanced' in file.lower():
                        model_type = 'Enhanced (54-D)'
                    else:
                        model_type = 'Neural Network (18-D)'
                else:
                    model_type = 'Unknown'

                models.append({
                    'name': file,
                    'size_kb': round(file_size, 2),
                    'type': model_type,
                    'active': f'backend/models/{file}' == current_model_name
                })

        # Sort: Model 4 first, then others
        def sort_key(x):
            if 'Model 4' in x['type']:
                return (0, x['name'])  # Model 4 first
            elif x['type'] == 'RandomForest':
                return (1, x['name'])  # RandomForest second
            else:
                return (2, x['name'])  # Others last
        
        models.sort(key=sort_key)

        return jsonify({
            'success': True,
            'models': models,
            'current_model': current_model_name
        })
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/set_model/<model_name>')
def set_model(model_name):
    """
    Switch to a different model dynamically
    Thread-safe model switching without restart
    """
    global classifier, current_model_name

    # Build full path: backend/models/model_name
    full_model_path = f'backend/models/{model_name}'
    model_path = os.path.join(PROJECT_ROOT, full_model_path)
    
    if not os.path.exists(model_path):
        return jsonify({
            'success': False,
            'error': f'Model file not found: {model_name}'
        }), 404

    if not (model_name.endswith('.pth') or model_name.endswith('.pkl')):
        return jsonify({
            'success': False,
            'error': 'Invalid model file (must be .pth or .pkl)'
        }), 400

    # Thread-safe model switching
    try:
        with model_lock:
            logger.info(f"🔄 Switching model: {current_model_name} → {model_name}")

            # Create new unified classifier with selected model
            new_classifier = UnifiedClassifier(model_path=model_path)

            # Test the new model with dummy input
            test_features = np.random.randn(18).astype(np.float32)
            _, _ = new_classifier.predict(test_features)

            # Switch to new model (atomic operation)
            classifier = new_classifier
            old_model = current_model_name
            current_model_name = full_model_path

            # Clear prediction buffers for fresh start
            prediction_buffer.clear()
            confidence_buffer.clear()

            logger.info(f"✅ Model switched successfully: {old_model} → {model_name}")

            # Broadcast to all connected clients
            socketio.emit('model_changed', {
                'model_name': model_name,
                'previous_model': old_model
            })

            return jsonify({
                'success': True,
                'model_name': model_name,
                'message': f'Model switched to {model_name}'
            })

    except Exception as e:
        logger.error(f"❌ Failed to switch model: {e}")
        return jsonify({
            'success': False,
            'error': f'Failed to load model: {str(e)}'
        }), 500


@app.route('/api/current_model')
def get_current_model():
    """Get currently active model info"""
    try:
        model_path = os.path.join(PROJECT_ROOT, current_model_name)
        file_size = os.path.getsize(model_path) / 1024 if os.path.exists(model_path) else 0

        return jsonify({
            'success': True,
            'model_name': current_model_name,
            'size_kb': round(file_size, 2),
            'path': model_path
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# Data Collection API Endpoints
@app.route('/api/collection_progress')
def get_collection_progress():
    """Get current collection progress"""
    if data_collector is None:
        return jsonify({'error': 'Data collector not initialized'}), 500
    
    return jsonify({
        'success': True,
        'progress': data_collector.samples_collected,
        'total': sum(data_collector.samples_collected.values()),
        'target_per_emote': data_collector.target_per_emote
    })


@app.route('/api/select_emote', methods=['POST'])
def select_emote():
    """Select emote for collection"""
    from flask import request
    data = request.json
    emote_id = data.get('emote_id', 0)
    
    if data_collector:
        data_collector.current_emote = emote_id
    
    return jsonify({'success': True, 'emote_id': emote_id})


@app.route('/api/start_collection', methods=['POST'])
def start_collection():
    """Start collecting data"""
    from flask import request
    data = request.json
    emote_id = data.get('emote_id', 0)
    
    if data_collector is None:
        return jsonify({'error': 'Data collector not initialized'}), 500
    
    data_collector.start_collection(emote_id)
    logger.info(f"Started collecting data for emote {emote_id}")
    
    return jsonify({'success': True, 'collecting': True})


@app.route('/api/stop_collection', methods=['POST'])
def stop_collection():
    """Stop collecting data"""
    if data_collector is None:
        return jsonify({'error': 'Data collector not initialized'}), 500
    
    data_collector.stop_collection()
    logger.info("Stopped data collection")
    
    return jsonify({'success': True, 'collecting': False})


@app.route('/api/export_collection', methods=['POST'])
def export_collection():
    """Export collected data"""
    if data_collector is None:
        return jsonify({'error': 'Data collector not initialized'}), 500
    
    result = data_collector.export_for_training()
    return jsonify(result)


# ========== Simple Data Capture API (New) ==========

def generate_capture_frames():
    """Generate video frames for simple capture interface"""
    if simple_collector is None or simple_collector.camera is None:
        logger.error("Simple collector or camera not initialized")
        return
    
    try:
        while True:
            frame = simple_collector.get_frame()
            if frame is None:
                logger.warning("Failed to get frame")
                time.sleep(0.1)
                continue
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    except Exception as e:
        logger.error(f"Error in generate_capture_frames: {e}")


@app.route('/capture_feed')
def capture_feed():
    """Video streaming route for capture page"""
    return Response(generate_capture_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/api/capture_progress')
def get_capture_progress():
    """Get current capture progress"""
    if simple_collector is None:
        return jsonify({'error': 'Simple collector not initialized'}), 500
    
    progress = simple_collector.get_progress()
    return jsonify({
        'success': True,
        **progress
    })


@app.route('/api/start_capture', methods=['POST'])
def start_capture():
    """Start capturing frames for selected emote"""
    from flask import request
    
    if simple_collector is None:
        return jsonify({'error': 'Simple collector not initialized'}), 500
    
    data = request.json
    emote_id = data.get('emote_id', 0)
    
    success = simple_collector.start_capture(emote_id)
    
    if success:
        # Emit camera info
        socketio.emit('camera_info', {
            'camera_index': simple_collector.active_camera_index
        })
        
        # Start progress monitoring thread
        def monitor_progress():
            import time
            while simple_collector.is_capturing:
                progress = simple_collector.get_progress()
                socketio.emit('capture_progress', progress)
                time.sleep(0.5)
            
            # Emit completion
            socketio.emit('capture_complete', {
                'emote_id': emote_id,
                'count': simple_collector.samples_collected[emote_id]
            })
        
        threading.Thread(target=monitor_progress, daemon=True).start()
        
        return jsonify({'success': True})
    else:
        return jsonify({'success': False, 'error': 'Failed to start capture'}), 500


@app.route('/api/stop_capture', methods=['POST'])
def stop_capture():
    """Stop capturing frames"""
    if simple_collector is None:
        return jsonify({'error': 'Simple collector not initialized'}), 500
    
    simple_collector.stop_capture()
    return jsonify({'success': True})


@app.route('/api/export_dataset', methods=['POST'])
def export_dataset():
    """Export captured dataset"""
    if simple_collector is None:
        return jsonify({'error': 'Simple collector not initialized'}), 500
    
    result = simple_collector.export_dataset()
    return jsonify({
        'success': True,
        **result
    })


# ========== ENHANCED COLLECTOR API (128-D Embeddings) ==========

@app.route('/api/enhanced/progress')
def get_enhanced_progress():
    """Get enhanced collector progress"""
    if enhanced_collector is None:
        return jsonify({'error': 'Enhanced collector not initialized'}), 500
    
    progress = enhanced_collector.get_progress()
    quality_stats = enhanced_collector.get_quality_stats()
    
    return jsonify({
        'success': True,
        'progress': progress,
        'quality': quality_stats
    })


@app.route('/api/enhanced/capture_sample', methods=['POST'])
def enhanced_capture_sample():
    """Capture a single sample with enhanced collector"""
    if enhanced_collector is None:
        return jsonify({'error': 'Enhanced collector not initialized'}), 500
    
    if camera is None:
        return jsonify({'error': 'Camera not initialized'}), 500
    
    data = request.json
    emote_id = data.get('emote_id', 0)
    
    # Get current frame
    with camera_lock:
        if camera is None or not camera.isOpened():
            return jsonify({'error': 'Camera not available'}), 500
        
        ret, frame = camera.read()
        if not ret or frame is None:
            return jsonify({'error': 'Failed to read frame'}), 500
    
    # Capture sample
    result = enhanced_collector.capture_sample(frame, emote_id)
    
    # Emit progress update
    if result['success']:
        progress = enhanced_collector.get_progress()
        socketio.emit('enhanced_progress', progress)
    
    return jsonify(result)


@app.route('/api/enhanced/reset_emote', methods=['POST'])
def reset_emote():
    """Reset collection for a specific emote"""
    if enhanced_collector is None:
        return jsonify({'error': 'Enhanced collector not initialized'}), 500
    
    data = request.json
    emote_id = data.get('emote_id')
    
    if emote_id is None:
        return jsonify({'error': 'emote_id required'}), 400
    
    result = enhanced_collector.reset_emote(emote_id)
    
    # Emit progress update
    if result['success']:
        progress = enhanced_collector.get_progress()
        socketio.emit('enhanced_progress', progress)
    
    return jsonify(result)


@app.route('/api/enhanced/reset_all', methods=['POST'])
def reset_all_data():
    """Reset all collected data"""
    if enhanced_collector is None:
        return jsonify({'error': 'Enhanced collector not initialized'}), 500
    
    result = enhanced_collector.reset_all()
    
    # Emit progress update
    if result['success']:
        progress = enhanced_collector.get_progress()
        socketio.emit('enhanced_progress', progress)
    
    return jsonify(result)


@app.route('/api/enhanced/export', methods=['POST'])
def export_enhanced_data():
    """Export enhanced training data"""
    if enhanced_collector is None:
        return jsonify({'error': 'Enhanced collector not initialized'}), 500
    
    result = enhanced_collector.export_training_data()
    
    if result['success']:
        return jsonify({
            'success': True,
            'message': result['message'],
            'num_samples': result['num_samples']
        })
    else:
        return jsonify(result), 400


# ========== Emote Management with Manifest ==========

def load_manifest():
    """Load emotes manifest from JSON file"""
    manifest_path = os.path.join(PROJECT_ROOT, 'emotes', 'manifest.json')
    try:
        if os.path.exists(manifest_path):
            with open(manifest_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Manifest not found at {manifest_path}, using defaults")
            return None
    except Exception as e:
        logger.error(f"Error loading manifest: {e}")
        return None

def save_manifest(manifest_data):
    """Save manifest to JSON file"""
    manifest_path = os.path.join(PROJECT_ROOT, 'emotes', 'manifest.json')
    try:
        os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
        with open(manifest_path, 'w') as f:
            json.dump(manifest_data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving manifest: {e}")
        return False

def load_label_map():
    """Load model label mapping"""
    map_path = os.path.join(PROJECT_ROOT, 'model_label_map.json')
    try:
        if os.path.exists(map_path):
            with open(map_path, 'r') as f:
                return json.load(f)
        return None
    except Exception as e:
        logger.error(f"Error loading label map: {e}")
        return None

# Load label mapping at startup
label_map = load_label_map()

@app.route('/api/emotes/list')
def list_emotes():
    """List all emotes with their training status from manifest"""
    try:
        manifest = load_manifest()
        
        if manifest and 'emotes' in manifest:
            # Use manifest data
            emotes = []
            for emote in manifest['emotes']:
                emotes.append({
                    'id': emote['id'],
                    'name': emote['id'],  # For backward compatibility
                    'label': emote['label'],
                    'display_name': emote['label'],
                    'filename': emote['image'],
                    'trained': emote.get('trained', False),
                    'audio': emote.get('audio', []),
                    'model_index': emote.get('model_index', -1)
                })
            
            return jsonify({
                'success': True,
                'emotes': emotes,
                'source': 'manifest'
            })
        else:
            # Fallback to scanning directory (legacy behavior)
            images_dir = os.path.join(PROJECT_ROOT, 'images')
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            
            # Get trained emotes from current model
            trained_emotes = set()
            if classifier and hasattr(classifier, 'model'):
                try:
                    if hasattr(classifier.model, 'emote_names'):
                        trained_emotes = set(classifier.model.emote_names.values())
                    elif hasattr(classifier.model, 'classes'):
                        trained_emotes = set(classifier.model.classes)
                except:
                    pass
            
            emotes = []
            for filename in os.listdir(images_dir):
                if filename.endswith(('.png', '.jpg', '.jpeg')):
                    name = os.path.splitext(filename)[0]
                    display_name = name.replace('_', ' ').title()
                    is_trained = name in trained_emotes or name.lower() in trained_emotes
                    
                    emotes.append({
                        'name': name,
                        'display_name': display_name,
                        'filename': filename,
                        'trained': is_trained
                    })
            
            emotes.sort(key=lambda x: (not x['trained'], x['display_name']))
            
            return jsonify({
                'success': True,
                'emotes': emotes,
                'source': 'directory_scan'
            })
    except Exception as e:
        logger.error(f"Error listing emotes: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/emotes/manifest')
def get_manifest():
    """Get complete manifest with audio configuration"""
    try:
        manifest = load_manifest()
        if manifest:
            return jsonify({
                'success': True,
                'manifest': manifest
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Manifest not found'
            }), 404
    except Exception as e:
        logger.error(f"Error getting manifest: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/emotes/update_training_status', methods=['POST'])
def update_training_status():
    """Update training status for an emote"""
    try:
        data = request.json
        emote_id = data.get('emote_id')
        trained = data.get('trained', False)
        
        if not emote_id:
            return jsonify({
                'success': False,
                'error': 'emote_id required'
            }), 400
        
        manifest = load_manifest()
        if not manifest or 'emotes' not in manifest:
            return jsonify({
                'success': False,
                'error': 'Manifest not found'
            }), 404
        
        # Find and update emote
        updated = False
        for emote in manifest['emotes']:
            if emote['id'] == emote_id:
                emote['trained'] = trained
                updated = True
                break
        
        if not updated:
            return jsonify({
                'success': False,
                'error': f'Emote {emote_id} not found'
            }), 404
        
        # Save manifest
        if save_manifest(manifest):
            return jsonify({
                'success': True,
                'message': f'Training status updated for {emote_id}'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to save manifest'
            }), 500
            
    except Exception as e:
        logger.error(f"Error updating training status: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/emotes/upload', methods=['POST'])
def upload_emote():
    """Upload new emote image"""
    try:
        from flask import request
        from werkzeug.utils import secure_filename
        
        # Check if image was uploaded
        if 'emote_image' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No image file provided'
            }), 400
        
        file = request.files['emote_image']
        emote_name = request.form.get('emote_name', '').strip()
        
        if not emote_name:
            return jsonify({
                'success': False,
                'error': 'Emote name is required'
            }), 400
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            }), 400
        
        # Validate file type
        allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
        file_ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        
        if file_ext not in allowed_extensions:
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(allowed_extensions)}'
            }), 400
        
        # Sanitize emote name
        emote_name = emote_name.lower().replace(' ', '_')
        emote_name = ''.join(c for c in emote_name if c.isalnum() or c == '_')
        
        # Create filename
        filename = f"{emote_name}.{file_ext}"
        
        # Save to images directory (in project root)
        images_dir = os.path.join(PROJECT_ROOT, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        
        filepath = os.path.join(images_dir, filename)
        
        # Check if file already exists
        if os.path.exists(filepath):
            return jsonify({
                'success': False,
                'error': f'Emote "{emote_name}" already exists'
            }), 400
        
        # Save file
        file.save(filepath)
        
        logger.info(f"✅ New emote uploaded: {emote_name} ({filename})")
        
        return jsonify({
            'success': True,
            'message': f'Emote "{emote_name}" added successfully!',
            'emote': {
                'name': emote_name,
                'filename': filename,
                'trained': False
            }
        })
        
    except Exception as e:
        logger.error(f"Error uploading emote: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        stats = {
            'model': {
                'name': current_model_name,
                'type': classifier.model_type if classifier else 'unknown',
                'input_dim': classifier.input_dim if classifier else 0
            },
            'emotes': {
                'total': 0,
                'trained': 0,
                'untrained': 0
            },
            'data': {
                'samples_collected': 0,
                'training_ready': False
            }
        }
        
        # Count emotes (in project root)
        images_dir = os.path.join(PROJECT_ROOT, 'images')
        if os.path.exists(images_dir):
            all_emotes = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
            stats['emotes']['total'] = len(all_emotes)
            
            # Get trained emotes
            if classifier and hasattr(classifier, 'model'):
                try:
                    if hasattr(classifier.model, 'emote_names'):
                        stats['emotes']['trained'] = len(classifier.model.emote_names)
                except:
                    pass
            
            stats['emotes']['untrained'] = stats['emotes']['total'] - stats['emotes']['trained']
        
        # Count training data
        pose_data_dir = os.path.join(PROJECT_ROOT, 'pose_data_v2')
        if os.path.exists(pose_data_dir):
            samples = [f for f in os.listdir(pose_data_dir) if f.endswith('.npz')]
            stats['data']['samples_collected'] = len(samples)
            stats['data']['training_ready'] = len(samples) >= 700  # 7 emotes * 100 samples
        
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    print("🚀 Starting Clash Emote Detector Web App...")
    print("📱 Open browser at: http://localhost:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
