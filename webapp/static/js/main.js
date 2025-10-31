/**
 * Clash Emote Detector+ - Frontend JavaScript
 * Rebuilt for camera detection and emote PNG display
 */

// Initialize Socket.IO
const socket = io();

// State
let cameraOn = true;
let lastEmote = null;
let audioEnabled = true;
let availableCameras = [];
let currentCameraIndex = 0;
let availableModels = [];
let currentModel = null;
let emoteManifest = null;

// Initialize new AudioManager - will be ready immediately
let audioManager = null;
let audioInitialized = false;

async function initAudio() {
    if (audioInitialized) return;
    
    try {
        audioManager = window.audioManager || new AudioManager();
        await audioManager.init();
        console.log('✅ Audio Manager initialized and ready');
        
        // Enable debug mode in development
        if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
            audioManager.enableDebug();
        }
        
        audioInitialized = true;
    } catch (error) {
        console.error('❌ Failed to initialize Audio Manager:', error);
    }
}

// Initialize audio immediately on page load (don't wait for click)
// Modern browsers allow this after user has interacted with site at least once
initAudio();

// Also try on first user interaction as fallback
document.addEventListener('click', () => {
    if (!audioInitialized) {
        console.log('🖱️ User clicked, initializing audio...');
        initAudio();
    }
}, { once: true });

// Retry initialization after short delay if failed
setTimeout(() => {
    if (!audioInitialized) {
        console.log('🔄 Retrying audio initialization...');
        initAudio();
    }
}, 1000);

// ========== Visual Feedback ==========

/**
 * Trigger visual feedback when audio plays successfully
 * Adds a glow effect to the current emote card
 */
function triggerAudioPlayedFeedback() {
    const currentEmoteCard = document.querySelector('#currentEmoteCard');
    if (!currentEmoteCard) return;
    
    // Add glow class
    currentEmoteCard.classList.add('audio-playing-glow');
    
    // Remove after animation (600ms)
    setTimeout(() => {
        currentEmoteCard.classList.remove('audio-playing-glow');
    }, 600);
}

// ========== Socket Events ==========

socket.on('connect', function() {
    console.log('✅ Connected to server');
});

socket.on('disconnect', function() {
    console.log('❌ Disconnected from server');
});

socket.on('emote_detected', async function(data) {
    const emote = data.emote;
    const confidence = data.confidence;
    
    console.log(`🎭 Emote detected: ${emote} (${(confidence * 100).toFixed(1)}%)`);
    
    // Update UI
    updateEmoteDisplay(emote, confidence);
    
    // Play sound for any valid emote (AudioManager handles debouncing)
    if (emote !== 'No Pose' && audioEnabled) {
        // Ensure AudioManager is initialized with retry
        if (!audioInitialized) {
            console.log('⏳ AudioManager not ready yet, initializing...');
            await initAudio();
            
            // Wait a bit more if still not ready
            if (!audioInitialized) {
                await new Promise(resolve => setTimeout(resolve, 100));
                await initAudio();
            }
        }
        
        if (audioManager && audioInitialized) {
            console.log(`🔊 Playing audio for: ${emote} (confidence: ${(confidence * 100).toFixed(1)}%)`);
            try {
                const played = await audioManager.playByLabel(emote, confidence);
                if (played) {
                    // Add visual feedback when audio plays successfully
                    triggerAudioPlayedFeedback();
                }
            } catch (err) {
                console.error('❌ Audio play error:', err);
            }
        } else {
            console.warn('⚠️ AudioManager still not available after retry');
        }
    }
    
    lastEmote = emote;
});

// ========== Camera Management ==========

async function loadAvailableCameras() {
    try {
        const response = await fetch('/api/list_cameras');
        const data = await response.json();
        
        if (data.success) {
            availableCameras = data.cameras;
            currentCameraIndex = data.current_index;
            updateCameraSelector();
            updateCameraStatus();
            console.log('📷 Cameras loaded:', availableCameras);
        } else {
            console.error('Failed to load cameras:', data.error);
            showNotification('Failed to load cameras', 'error');
            updateCameraStatus('Error');
        }
    } catch (error) {
        console.error('Error loading cameras:', error);
        showNotification('Error loading cameras', 'error');
        updateCameraStatus('Error');
    }
}

function updateCameraStatus(status = null) {
    const statusEl = document.getElementById('cameraStatus');
    if (!statusEl) return;
    
    if (status) {
        statusEl.textContent = status;
        return;
    }
    
    // Auto-detect status based on available cameras
    const currentCamera = availableCameras.find(c => c.index === currentCameraIndex);
    if (currentCamera) {
        statusEl.textContent = currentCamera.name === 'Laptop Camera' ? '💻' : '📱';
    } else {
        statusEl.textContent = 'Ready';
    }
}

function updateCameraSelector() {
    const selector = document.getElementById('cameraSelector');
    if (!selector) return;
    
    if (availableCameras.length === 0) {
        selector.innerHTML = '<option value="">No cameras found</option>';
        return;
    }
    
    selector.innerHTML = availableCameras.map(camera => {
        const isActive = camera.index === currentCameraIndex;
        // Use emoji based on camera type
        const emoji = camera.name === 'Laptop Camera' ? '💻' : '📱';
        return `<option value="${camera.index}" ${isActive ? 'selected' : ''}>
            ${emoji} ${camera.name}
        </option>`;
    }).join('');
}

async function switchCamera(cameraIndex) {
    try {
        console.log(`🔄 Switching to camera ${cameraIndex}...`);
        updateCameraStatus('Switching...');
        
        const response = await fetch('/api/switch_camera', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ camera_index: parseInt(cameraIndex) })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentCameraIndex = data.camera_index;
            console.log(`✅ Switched to camera ${currentCameraIndex}`);
            
            // Get camera name for notification
            const camera = availableCameras.find(c => c.index === currentCameraIndex);
            const cameraName = camera ? camera.name : `Camera ${currentCameraIndex}`;
            showNotification(`Switched to ${cameraName}`, 'success');
            updateCameraStatus();
            
            // Reload video feed with timestamp to avoid caching
            const videoFeed = document.getElementById('videoFeed');
            if (videoFeed) {
                const timestamp = new Date().getTime();
                videoFeed.src = `/video_feed?t=${timestamp}`;
            }
        } else {
            console.error('Failed to switch camera:', data.error);
            showNotification('Failed to switch camera', 'error');
            updateCameraStatus('Error');
        }
    } catch (error) {
        console.error('Error switching camera:', error);
        showNotification('Error switching camera', 'error');
        updateCameraStatus('Error');
    }
}

// Camera selector event listener
const cameraSelector = document.getElementById('cameraSelector');
if (cameraSelector) {
    cameraSelector.addEventListener('change', (e) => {
        const cameraIndex = parseInt(e.target.value);
        if (!isNaN(cameraIndex)) {
            switchCamera(cameraIndex);
        }
    });
}

// ========== Camera Toggle ==========

const toggleCameraBtn = document.getElementById('toggleCamera');
if (toggleCameraBtn) {
    toggleCameraBtn.addEventListener('click', () => {
        cameraOn = !cameraOn;
        socket.emit('toggle_camera', { active: cameraOn });
        
        if (cameraOn) {
            toggleCameraBtn.textContent = '🎥 Camera ON';
            toggleCameraBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
            toggleCameraBtn.classList.add('bg-green-600', 'hover:bg-green-700');
        } else {
            toggleCameraBtn.textContent = '📷 Camera OFF';
            toggleCameraBtn.classList.remove('bg-green-600', 'hover:bg-green-700');
            toggleCameraBtn.classList.add('bg-red-600', 'hover:bg-red-700');
        }
        
        console.log(`📷 Camera ${cameraOn ? 'ON' : 'OFF'}`);
    });
}

// ========== Emote Display ==========

let manifestLoadAttempts = 0;
const MAX_MANIFEST_ATTEMPTS = 3;

async function loadManifest() {
    if (emoteManifest) return emoteManifest;
    
    // Prevent infinite retry spam
    if (manifestLoadAttempts >= MAX_MANIFEST_ATTEMPTS) {
        console.warn('⚠️ Max manifest load attempts reached, using fallback mode');
        return null;
    }
    
    manifestLoadAttempts++;
    
    try {
        const response = await fetch('/api/emotes/manifest');
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        if (data.success && data.manifest) {
            emoteManifest = data.manifest;
            console.log('✅ Manifest loaded:', emoteManifest);
            manifestLoadAttempts = 0; // Reset on success
            return emoteManifest;
        } else {
            throw new Error('Invalid manifest response');
        }
    } catch (error) {
        console.error(`❌ Failed to load manifest (attempt ${manifestLoadAttempts}/${MAX_MANIFEST_ATTEMPTS}):`, error.message);
    }
    return null;
}

function getEmoteFromManifest(emoteName) {
    if (!emoteManifest || !emoteManifest.emotes) return null;
    
    // Try exact label match first
    let emote = emoteManifest.emotes.find(e => e.label === emoteName);
    if (emote) return emote;
    
    // Try case-insensitive label match
    emote = emoteManifest.emotes.find(e => 
        e.label.toLowerCase() === emoteName.toLowerCase()
    );
    if (emote) return emote;
    
    // Normalize emoteName for legacy label matching (replace underscores with spaces)
    const normalizedName = emoteName.replace(/_/g, ' ');
    
    // Try legacy label match with normalized name
    emote = emoteManifest.emotes.find(e => 
        e.legacy_label && e.legacy_label.toLowerCase() === normalizedName.toLowerCase()
    );
    if (emote) return emote;
    
    // Try ID match
    const emoteId = emoteName.toLowerCase().replace(/\s+/g, '_');
    emote = emoteManifest.emotes.find(e => e.id === emoteId);
    
    return emote;
}

async function updateEmoteDisplay(emote, confidence) {
    // Ensure manifest is loaded
    if (!emoteManifest) {
        await loadManifest();
    }
    
    // Update stats
    const emoteTextEl = document.getElementById('emoteText');
    const confidenceEl = document.getElementById('confidence');
    
    if (emoteTextEl) {
        emoteTextEl.textContent = emote;
    }
    
    if (confidenceEl) {
        confidenceEl.textContent = `${(confidence * 100).toFixed(0)}%`;
    }
    
    // Update current emote card
    const currentEmoteImage = document.getElementById('currentEmoteImage');
    const currentEmoteEmoji = document.getElementById('currentEmoteEmoji');
    const currentEmoteName = document.getElementById('currentEmoteName');
    const currentEmoteConfidence = document.getElementById('currentEmoteConfidence');
    
    if (emote === 'No Pose') {
        // Show emoji placeholder
        if (currentEmoteImage) currentEmoteImage.classList.add('hidden');
        if (currentEmoteEmoji) {
            currentEmoteEmoji.classList.remove('hidden');
            currentEmoteEmoji.textContent = '🎭';
        }
        if (currentEmoteName) currentEmoteName.textContent = 'No Pose';
        if (currentEmoteConfidence) currentEmoteConfidence.textContent = 'Waiting for detection...';
    } else {
        // Get emote from manifest
        const emoteInfo = getEmoteFromManifest(emote);
        
        if (emoteInfo) {
            // Show emote PNG from manifest
            if (currentEmoteImage && emoteInfo.image) {
                currentEmoteImage.src = `/images/${emoteInfo.image}`;
                currentEmoteImage.classList.remove('hidden');
            }
            if (currentEmoteEmoji) currentEmoteEmoji.classList.add('hidden');
            if (currentEmoteName) currentEmoteName.textContent = emoteInfo.label;
            if (currentEmoteConfidence) {
                currentEmoteConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
            }
        } else {
            // Fallback if emote not in manifest
            console.warn(`⚠️ Emote not found in manifest: ${emote}`);
            if (currentEmoteImage) currentEmoteImage.classList.add('hidden');
            if (currentEmoteEmoji) {
                currentEmoteEmoji.classList.remove('hidden');
                currentEmoteEmoji.textContent = '❓';
            }
            if (currentEmoteName) currentEmoteName.textContent = emote;
            if (currentEmoteConfidence) {
                currentEmoteConfidence.textContent = `Confidence: ${(confidence * 100).toFixed(1)}%`;
            }
        }
    }
}

// Audio toggle using AudioManager
function toggleAudio() {
    if (audioManager) {
        audioEnabled = audioManager.toggle();
    } else {
        audioEnabled = !audioEnabled;
    }
    return audioEnabled;
}

// ========== Notifications ==========

function showNotification(message, type = 'info') {
    const colors = {
        success: 'bg-green-600',
        error: 'bg-red-600',
        warning: 'bg-yellow-600',
        info: 'bg-blue-600'
    };
    
    const notification = document.createElement('div');
    notification.className = `${colors[type]} text-white px-6 py-4 rounded-xl shadow-2xl transition-all`;
    notification.textContent = message;
    
    const container = document.getElementById('notificationContainer');
    if (container) {
        container.appendChild(notification);
        
        setTimeout(() => {
            notification.style.opacity = '0';
            notification.style.transform = 'translateY(-10px)';
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// ========== FPS Counter ==========

let frameCount = 0;
let lastFpsUpdate = Date.now();

function updateFPS() {
    frameCount++;
    const now = Date.now();
    const elapsed = now - lastFpsUpdate;
    
    if (elapsed >= 1000) {
        const fps = Math.round((frameCount * 1000) / elapsed);
        const fpsEl = document.getElementById('fps');
        if (fpsEl) {
            fpsEl.textContent = fps;
        }
        frameCount = 0;
        lastFpsUpdate = now;
    }
}

// Update FPS every frame
setInterval(updateFPS, 100);

// ========== Model Management ==========

async function loadAvailableModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        
        if (data.success) {
            availableModels = data.models;
            currentModel = data.current_model;
            updateModelSelector();
            console.log('🤖 Models loaded:', availableModels);
        } else {
            console.error('Failed to load models:', data.error);
            showNotification('Failed to load models', 'error');
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showNotification('Error loading models', 'error');
    }
}

function updateModelSelector() {
    const selector = document.getElementById('modelSelector');
    if (!selector) return;
    
    if (availableModels.length === 0) {
        selector.innerHTML = '<option value="">No models found</option>';
        return;
    }
    
    // Sort models by priority
    const sortedModels = [...availableModels].sort((a, b) => {
        // Priority order: Model 4 > Embedding > Enhanced > Neural > RandomForest
        const priority = {
            '🚀 Model 4 Ultimate (128-D)': 0,
            'Embedding (128-D)': 1,
            'Enhanced (54-D)': 2,
            'Neural Network (18-D)': 3,
            'RandomForest': 4
        };
        return (priority[a.type] || 99) - (priority[b.type] || 99);
    });
    
    let html = '';
    let modelCounter = { rf: 0, nn: 0, emb: 0, enh: 0, m4: 0 };
    
    sortedModels.forEach(model => {
        const isActive = model.name === currentModel;
        let label = '';
        let emoji = '';
        
        // Smart naming based on model type
        if (model.type.includes('Model 4')) {
            emoji = '🚀';
            label = `${emoji} Model 3 Ultra (Best - 98.57%)`;
        } else if (model.type.includes('Embedding')) {
            emoji = '🧬';
            label = `${emoji} Model 2 (Advanced)`;
        } else if (model.type.includes('Enhanced')) {
            emoji = '⚡';
            label = `${emoji} Model 1 (Good)`;
        } else if (model.type === 'RandomForest') {
            emoji = '🌲';
            label = `${emoji} Model 0 (Legacy)`;
        } else {
            // Skip other neural network models to avoid duplicates
            return;
        }
        
        html += `<option value="${model.name}" ${isActive ? 'selected' : ''}>${label}</option>`;
        
        // Update current model display
        if (isActive) {
            const modelDisplay = document.getElementById('currentModelDisplay');
            if (modelDisplay) {
                modelDisplay.textContent = label;
            }
        }
    });
    
    selector.innerHTML = html;
}

async function switchModel(modelName) {
    try {
        console.log(`🔄 Switching to model: ${modelName}...`);
        showNotification('Switching model...', 'info');
        
        const response = await fetch(`/api/set_model/${modelName}`);
        const data = await response.json();
        
        if (data.success) {
            currentModel = data.model_name;
            console.log(`✅ Switched to model: ${currentModel}`);
            showNotification(`Model switched to ${modelName}`, 'success');
        } else {
            console.error('Failed to switch model:', data.error);
            showNotification(`Failed to switch model: ${data.error}`, 'error');
            // Revert selector to current model
            updateModelSelector();
        }
    } catch (error) {
        console.error('Error switching model:', error);
        showNotification('Error switching model', 'error');
        // Revert selector to current model
        updateModelSelector();
    }
}

// Model selector event listener
const modelSelector = document.getElementById('modelSelector');
if (modelSelector) {
    modelSelector.addEventListener('change', (e) => {
        const modelName = e.target.value;
        if (modelName) {
            switchModel(modelName);
        }
    });
}

// Listen for model changes from server
socket.on('model_changed', function(data) {
    console.log(`🔄 Model changed: ${data.previous_model} → ${data.model_name}`);
    currentModel = data.model_name;
    updateModelSelector();
    showNotification(`Model changed to ${data.model_name}`, 'info');
});

// ========== Keyboard Shortcuts ==========

function toggleHelp() {
    const helpPanel = document.getElementById('helpPanel');
    if (helpPanel) {
        helpPanel.classList.toggle('hidden');
        helpPanel.classList.toggle('flex');
    }
}

// Keyboard event handler
document.addEventListener('keydown', (e) => {
    // Ignore if typing in input/select
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') {
        return;
    }
    
    switch(e.key.toLowerCase()) {
        case ' ':  // Space - Toggle camera
            e.preventDefault();
            toggleCamera();
            break;
            
        case 'm':  // M - Mute/unmute audio
            e.preventDefault();
            const enabled = toggleAudio();
            showNotification(`Audio ${enabled ? 'enabled' : 'disabled'}`, 'info');
            break;
            
        case 'c':  // C - Switch camera
            e.preventDefault();
            const cameraSelector = document.getElementById('cameraSelector');
            if (cameraSelector && availableCameras.length > 1) {
                const nextIndex = (currentCameraIndex + 1) % availableCameras.length;
                cameraSelector.selectedIndex = nextIndex;
                cameraSelector.dispatchEvent(new Event('change'));
            }
            break;
            
        case '?':  // ? - Show help
            e.preventDefault();
            toggleHelp();
            break;
            
        case 'escape':  // Esc - Close help
            const helpPanel = document.getElementById('helpPanel');
            if (helpPanel && !helpPanel.classList.contains('hidden')) {
                toggleHelp();
            }
            break;
            
        case '1':  // Quick switch to Model 1
        case '2':  // Quick switch to Model 2
        case '3':  // Quick switch to Model 3
        case '4':  // Quick switch to Model 4
            e.preventDefault();
            const modelIndex = parseInt(e.key) - 1;
            if (availableModels[modelIndex]) {
                switchModel(availableModels[modelIndex].name);
            }
            break;
    }
});

// ========== Emotes Grid ==========

async function loadEmotesGrid() {
    try {
        // Add cache busting to ensure fresh data
        const timestamp = new Date().getTime();
        const response = await fetch(`/api/emotes/list?_=${timestamp}`);
        const data = await response.json();
        
        console.log('📋 Loaded emotes from API:', data);
        
        const grid = document.getElementById('emotesGrid');
        if (!grid) return;
        
        if (data.success && data.emotes && data.emotes.length > 0) {
            const colors = [
                'from-yellow-600/20 to-orange-700/20 border-yellow-600/50',
                'from-blue-600/20 to-purple-700/20 border-blue-600/50',
                'from-blue-400/20 to-cyan-700/20 border-blue-400/50',
                'from-purple-600/20 to-pink-700/20 border-purple-600/50',
                'from-red-600/20 to-orange-700/20 border-red-600/50',
                'from-pink-600/20 to-rose-700/20 border-pink-600/50',
                'from-green-600/20 to-emerald-700/20 border-green-600/50',
                'from-indigo-600/20 to-purple-700/20 border-indigo-600/50'
            ];
            
            grid.innerHTML = data.emotes.map((emote, index) => {
                const colorClass = colors[index % colors.length];
                const isTrained = emote.trained;
                
                // Trained: normal appearance (no badge)
                // Untrained: grayed out + red cross
                const cardClass = isTrained 
                    ? `bg-gradient-to-br ${colorClass} hover:scale-105 cursor-pointer`
                    : `bg-gray-700/30 border-gray-600/50 opacity-60 cursor-not-allowed`;
                
                const statusBadge = isTrained 
                    ? ''  // No badge for trained
                    : '<span class="text-xs bg-red-600 px-2 py-1 rounded ml-1" title="Untrained — perform training to enable">✗</span>';
                
                const imgClass = isTrained ? '' : 'grayscale opacity-50';
                const tooltip = isTrained ? emote.display_name : 'Untrained — perform training to enable';
                const ariaDisabled = isTrained ? '' : 'aria-disabled="true"';
                const tabindex = isTrained ? '0' : '-1';
                
                return `
                    <div class="${cardClass} rounded-xl p-3 text-center transition transform relative" 
                         title="${tooltip}" 
                         ${ariaDisabled}
                         tabindex="${tabindex}"
                         role="button">
                        <img src="/images/${emote.filename}" 
                             alt="${tooltip}" 
                             class="w-16 h-16 mx-auto mb-2 object-contain rounded ${imgClass}"
                             onerror="this.src='data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 width=%22100%22 height=%22100%22%3E%3Crect fill=%22%23444%22 width=%22100%22 height=%22100%22/%3E%3Ctext fill=%22%23fff%22 font-size=%2214%22 x=%2250%25%22 y=%2250%25%22 text-anchor=%22middle%22%3E${emote.display_name[0]}%3C/text%3E%3C/svg%3E'">
                        <div class="text-xs font-semibold flex items-center justify-center">
                            ${emote.display_name}
                            ${statusBadge}
                        </div>
                    </div>
                `;
            }).join('');
        } else {
            grid.innerHTML = '<div class="col-span-2 text-center py-8 text-gray-400">No emotes found</div>';
        }
    } catch (error) {
        console.error('Error loading emotes grid:', error);
        const grid = document.getElementById('emotesGrid');
        if (grid) {
            grid.innerHTML = '<div class="col-span-2 text-center py-8 text-red-400">Error loading emotes</div>';
        }
    }
}

// ========== Video Feed Management ==========

function initVideoFeed() {
    const videoFeed = document.getElementById('videoFeed');
    if (!videoFeed) return;
    
    console.log('🎥 Initializing video feed...');
    console.log('📍 Video feed URL:', videoFeed.src);
    
    // Force reload after a short delay to ensure backend is ready
    setTimeout(() => {
        const timestamp = new Date().getTime();
        const newSrc = `/video_feed?t=${timestamp}`;
        console.log('🔄 Reloading video feed:', newSrc);
        videoFeed.src = newSrc;
    }, 1000);
    
    // Check if video is loading
    videoFeed.addEventListener('load', () => {
        console.log('✅ Video feed image loaded');
    });
    
    videoFeed.addEventListener('error', (e) => {
        console.error('❌ Video feed error:', e);
        console.error('Failed URL:', videoFeed.src);
    });
}

// ========== Initialization ==========

// Load manifest first (needed for everything)
loadManifest().then(() => {
    console.log('✅ Manifest loaded on startup');
});

// Load cameras and models on page load
loadAvailableCameras();
loadAvailableModels();
loadEmotesGrid();

// Initialize video feed
initVideoFeed();

// Initialize audio immediately
setTimeout(() => {
    initAudio();
    console.log('🔊 Audio auto-initialized');
}, 500);

// Show welcome notification with keyboard shortcut hint
setTimeout(() => {
    showNotification('💡 Press ? for keyboard shortcuts', 'info');
}, 2000);

console.log('🚀 Clash Emote Detector+ initialized');
console.log('📷 Camera detection enabled');
console.log('🎭 Emote PNG display enabled');
console.log('🤖 Model selection enabled');
