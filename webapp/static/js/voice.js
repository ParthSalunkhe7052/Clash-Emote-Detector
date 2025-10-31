/**
 * Voice Guesser - Frontend JavaScript
 * Author: Parth
 */

const socket = io();
let isRecording = false;
let mediaRecorder = null;
let audioChunks = [];

// Elements
const recordBtn = document.getElementById('recordBtn');
const voiceStatus = document.getElementById('voiceStatus');
const waveformCanvas = document.getElementById('waveform');
const waveformCtx = waveformCanvas ? waveformCanvas.getContext('2d') : null;

// Socket events
socket.on('connect', function() {
    console.log('✅ Connected to voice server');
});

socket.on('voice_detected', function(data) {
    console.log('Voice detected:', data);
    updateVoiceDisplay(data.emote, data.confidence);
});

// Record button handler
if (recordBtn) {
    recordBtn.addEventListener('click', async function() {
        if (!isRecording) {
            await startRecording();
        } else {
            stopRecording();
        }
    });
}

async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = function(event) {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = function() {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            sendAudioToServer(audioBlob);
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        recordBtn.textContent = '⏹️ Recording...';
        recordBtn.classList.remove('bg-red-600', 'hover:bg-red-700');
        recordBtn.classList.add('bg-green-600', 'hover:bg-green-700', 'animate-pulse');
        voiceStatus.textContent = 'Recording... Make your emote sound!';
        
        // Auto-stop after 2.5 seconds
        setTimeout(() => {
            if (isRecording) {
                stopRecording();
            }
        }, 2500);
        
    } catch (error) {
        console.error('Microphone access denied:', error);
        voiceStatus.textContent = '❌ Microphone access denied. Please allow microphone.';
    }
}

function stopRecording() {
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        mediaRecorder.stream.getTracks().forEach(track => track.stop());
    }
    
    isRecording = false;
    recordBtn.innerHTML = '<span class="text-2xl">🎙️</span><span>Start Recording</span>';
    recordBtn.classList.remove('bg-green-600', 'hover:bg-green-700', 'animate-pulse');
    recordBtn.classList.add('bg-red-600', 'hover:bg-red-700');
    voiceStatus.textContent = 'Processing...';
}

function sendAudioToServer(audioBlob) {
    // For now, just show a message
    // TODO: Implement server-side voice classification
    voiceStatus.textContent = '⚠️ Voice model not trained yet. This is a preview.';
    
    console.log('Audio recorded:', audioBlob.size, 'bytes');
    
    // Simulate detection after 1 second
    setTimeout(() => {
        voiceStatus.textContent = 'Ready to record';
    }, 2000);
}

function updateVoiceDisplay(emote, confidence) {
    // Update display when voice detection is implemented
    console.log('Voice emote:', emote, 'Confidence:', confidence);
}

// Draw waveform placeholder
if (waveformCtx) {
    waveformCanvas.width = waveformCanvas.offsetWidth;
    waveformCanvas.height = waveformCanvas.offsetHeight;
}

console.log('🎤 Voice guesser initialized');
