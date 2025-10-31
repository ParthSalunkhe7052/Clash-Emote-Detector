/**
 * Robust Audio Manager for Clash Emote Detector
 * Handles audio playback with debouncing, preloading, and fallback
 */

class AudioManager {
    constructor() {
        this.manifest = null;
        this.audioConfig = {
            confidence_threshold: 0.65,
            debounce_ms: 1500,
            preload_count: 6,
            default_missing_audio: '/sounds/default_missing_audio.mp3',
            max_concurrent_players: 1,
            stop_previous_on_new: true,
            format_priority: ['ogg', 'mp3', 'wav']
        };
        
        // Audio state
        this.audioCache = new Map();  // emote_id -> HTMLAudioElement
        this.lastPlayedTime = new Map();  // emote_id -> timestamp
        this.currentPlayer = null;
        this.enabled = true;
        this.initialized = false;
        
        // Developer logging toggle
        this.debugMode = false;
        
        // Statistics
        this.stats = {
            played: 0,
            debounced: 0,
            missing: 0,
            failed: 0
        };
    }
    
    /**
     * Initialize audio manager and load manifest
     */
    async init() {
        if (this.initialized) {
            this.log('✅ Already initialized');
            return true;
        }
        
        this.log('🎵 Initializing Audio Manager...');
        
        try {
            // Load manifest with timeout
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);
            
            const response = await fetch('/api/emotes/manifest', {
                signal: controller.signal
            });
            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            
            if (data.success && data.manifest) {
                this.manifest = data.manifest;
                this.audioConfig = { ...this.audioConfig, ...data.manifest.audio_config };
                this.log('✅ Manifest loaded:', this.manifest);
                
                // Preload top N trained emotes
                await this.preloadAudio();
                
                // Only mark as initialized after everything is ready
                this.initialized = true;
                this.log('✅ Audio Manager fully initialized and ready');
                return true;
            } else {
                console.error('❌ Failed to load manifest: Invalid response', data);
                this.initialized = false;
                return false;
            }
        } catch (error) {
            console.error('❌ Error initializing Audio Manager:', error.message);
            this.initialized = false;
            return false;
            // Don't throw - allow app to continue without audio
        }
    }
    
    /**
     * Preload audio for top N trained emotes
     */
    async preloadAudio() {
        if (!this.manifest || !this.manifest.emotes) return;
        
        const trainedEmotes = this.manifest.emotes
            .filter(e => e.trained)
            .slice(0, this.audioConfig.preload_count);
        
        this.log(`📦 Preloading ${trainedEmotes.length} emote audio files...`);
        
        for (const emote of trainedEmotes) {
            await this.loadAudioForEmote(emote.id);
        }
        
        this.log(`✅ Preloaded ${trainedEmotes.length} emote audio files`);
    }
    
    /**
     * Load audio element for an emote with fallback support
     */
    async loadAudioForEmote(emoteId) {
        // Check if already cached
        if (this.audioCache.has(emoteId)) {
            return this.audioCache.get(emoteId);
        }
        
        const emote = this.manifest.emotes.find(e => e.id === emoteId);
        if (!emote) {
            this.log(`⚠️ Emote ${emoteId} not found in manifest`);
            return null;
        }
        
        // Try each audio file in priority order
        for (const audioFile of emote.audio) {
            const audioPath = `/sounds/${audioFile}`;
            
            try {
                const audio = new Audio();
                audio.preload = 'auto';
                
                // Wait for audio to be ready
                const loaded = await new Promise((resolve) => {
                    audio.addEventListener('canplaythrough', () => resolve(true), { once: true });
                    audio.addEventListener('error', () => resolve(false), { once: true });
                    audio.src = audioPath;
                });
                
                if (loaded) {
                    this.log(`✅ Loaded audio: ${audioPath}`);
                    this.audioCache.set(emoteId, audio);
                    return audio;
                }
            } catch (error) {
                this.log(`⚠️ Failed to load ${audioPath}: ${error.message}`);
            }
        }
        
        // All formats failed, try default missing audio
        this.log(`❌ No audio found for ${emoteId}, using default`);
        this.stats.missing++;
        
        try {
            const audio = new Audio(this.audioConfig.default_missing_audio);
            audio.preload = 'auto';
            this.audioCache.set(emoteId, audio);
            return audio;
        } catch (error) {
            this.log(`❌ Even default audio failed: ${error.message}`);
            return null;
        }
    }
    
    /**
     * Calculate volume from confidence (dynamic scaling)
     * - confidence >= 0.6: full volume (1.0)
     * - confidence 0.45-0.6: scaled volume (linear)
     * - confidence < 0.45: skip audio (return 0)
     */
    getVolumeFromConfidence(confidence) {
        if (confidence < 0.45) {
            return 0;  // Skip audio
        }
        if (confidence >= 0.6) {
            return 1.0;  // Full volume
        }
        // Linear scale between 0.45 and 0.6
        return Math.min(1, Math.max(0, (confidence - 0.45) / 0.15));
    }
    
    /**
     * Play audio for detected emote with dynamic volume and debounce
     */
    async playEmote(emoteId, confidence) {
        if (!this.initialized) {
            const success = await this.init();
            if (!success) {
                console.error('❌ Cannot play audio: AudioManager initialization failed');
                return false;
            }
        }
        
        // Double check manifest is loaded
        if (!this.manifest || !this.manifest.emotes) {
            console.error('❌ Cannot play audio: Manifest not loaded');
            return false;
        }
        
        if (!this.enabled) {
            this.log('🔇 Audio disabled');
            return false;
        }
        
        // Calculate volume from confidence (new dynamic system)
        const volume = this.getVolumeFromConfidence(confidence);
        if (volume === 0) {
            this.log(`⏭️ Confidence ${confidence.toFixed(2)} below minimum threshold (0.45)`);
            return false;
        }
        
        // Check debounce (per-emote cooldown)
        const now = Date.now();
        const lastPlayed = this.lastPlayedTime.get(emoteId);
        
        if (lastPlayed && (now - lastPlayed) < this.audioConfig.debounce_ms) {
            this.log(`⏭️ Cooldown active for ${emoteId}, skipping (${now - lastPlayed}ms ago)`);
            this.stats.debounced++;
            return false;
        }
        
        // Find emote in manifest
        const emote = this.manifest.emotes.find(e => e.id === emoteId);
        if (!emote) {
            this.log(`⚠️ Emote ${emoteId} not in manifest`);
            return false;
        }
        
        // Check if trained
        if (!emote.trained) {
            this.log(`⏭️ Emote ${emoteId} is untrained, skipping audio`);
            return false;
        }
        
        // Get or load audio
        let audio = this.audioCache.get(emoteId);
        if (!audio) {
            audio = await this.loadAudioForEmote(emoteId);
        }
        
        if (!audio) {
            this.log(`❌ No audio available for ${emoteId}`);
            this.stats.failed++;
            return false;
        }
        
        // Stop previous audio if configured
        if (this.audioConfig.stop_previous_on_new && this.currentPlayer) {
            this.currentPlayer.pause();
            this.currentPlayer.currentTime = 0;
        }
        
        // Play audio with dynamic volume
        try {
            audio.currentTime = 0;
            audio.volume = volume;  // Set dynamic volume based on confidence
            await audio.play();
            
            this.currentPlayer = audio;
            this.lastPlayedTime.set(emoteId, now);
            this.stats.played++;
            
            this.log(`🔊 Playing: ${emoteId} (${emote.label}) - volume: ${volume.toFixed(2)} (confidence: ${confidence.toFixed(2)})`);
            return true;
        } catch (error) {
            this.log(`❌ Play failed for ${emoteId}: ${error.message}`);
            this.stats.failed++;
            return false;
        }
    }
    
    /**
     * Play audio by label (with label mapping support)
     */
    async playByLabel(label, confidence) {
        if (!this.manifest) {
            await this.init();
        }
        
        // Normalize label (replace underscores with spaces for legacy matching)
        const normalizedLabel = label.replace(/_/g, ' ');
        
        // Find emote by label with comprehensive matching
        const emote = this.manifest.emotes.find(e => 
            e.label === label || 
            e.label.toLowerCase() === label.toLowerCase() ||
            e.legacy_label === label ||
            (e.legacy_label && e.legacy_label.toLowerCase() === normalizedLabel.toLowerCase()) ||
            e.id === label.toLowerCase().replace(/\s+/g, '_')
        );
        
        if (emote) {
            return await this.playEmote(emote.id, confidence);
        } else {
            this.log(`⚠️ No emote found for label: ${label}`);
            return false;
        }
    }
    
    /**
     * Enable audio playback
     */
    enable() {
        this.enabled = true;
        this.log('🔊 Audio enabled');
    }
    
    /**
     * Disable audio playback
     */
    disable() {
        this.enabled = false;
        this.log('🔇 Audio disabled');
    }
    
    /**
     * Toggle audio playback
     */
    toggle() {
        this.enabled = !this.enabled;
        this.log(this.enabled ? '🔊 Audio enabled' : '🔇 Audio disabled');
        return this.enabled;
    }
    
    /**
     * Enable debug logging
     */
    enableDebug() {
        this.debugMode = true;
        console.log('🐛 Audio debug mode enabled');
    }
    
    /**
     * Disable debug logging
     */
    disableDebug() {
        this.debugMode = false;
    }
    
    /**
     * Get statistics
     */
    getStats() {
        return { ...this.stats };
    }
    
    /**
     * Reset statistics
     */
    resetStats() {
        this.stats = {
            played: 0,
            debounced: 0,
            missing: 0,
            failed: 0
        };
        this.log('📊 Stats reset');
    }
    
    /**
     * Log with debug mode check
     */
    log(...args) {
        if (this.debugMode) {
            console.log('[AudioManager]', ...args);
        }
    }
    
    /**
     * Validate manifest and audio files
     */
    async validate() {
        if (!this.manifest) {
            console.error('❌ No manifest loaded');
            return false;
        }
        
        console.log('🔍 Validating manifest and audio files...');
        
        const results = {
            total: 0,
            trained: 0,
            valid: 0,
            missing: 0,
            errors: []
        };
        
        for (const emote of this.manifest.emotes) {
            results.total++;
            
            if (emote.trained) {
                results.trained++;
                
                // Try to load audio
                const audio = await this.loadAudioForEmote(emote.id);
                if (audio && audio.src !== this.audioConfig.default_missing_audio) {
                    results.valid++;
                    console.log(`✅ ${emote.label}: Audio OK`);
                } else {
                    results.missing++;
                    results.errors.push(`${emote.label}: No audio file found (expected one of: ${emote.audio.join(', ')})`);
                    console.error(`❌ ${emote.label}: MISSING AUDIO`);
                }
            } else {
                console.log(`⏭️ ${emote.label}: Untrained (skipped)`);
            }
        }
        
        console.log('\n📊 Validation Results:');
        console.log(`   Total emotes: ${results.total}`);
        console.log(`   Trained: ${results.trained}`);
        console.log(`   Valid audio: ${results.valid}`);
        console.log(`   Missing audio: ${results.missing}`);
        
        if (results.errors.length > 0) {
            console.error('\n❌ Errors:');
            results.errors.forEach(err => console.error(`   - ${err}`));
        }
        
        return results.missing === 0;
    }
}

// Export global instance
window.audioManager = new AudioManager();
