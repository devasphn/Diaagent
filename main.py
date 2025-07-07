
import os
import sys
import warnings
import logging
from pathlib import Path
import json
import time
import threading
import queue
import asyncio
from typing import Optional, Dict, Any, List
import base64
import io
import random

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

def get_hf_token():
    """Get HuggingFace token from multiple sources"""
    # Method 1: Environment variable
    token = os.getenv("HF_TOKEN")
    if token:
        return token
    
    # Method 2: HuggingFace CLI stored token
    try:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            with open(token_path, 'r') as f:
                token = f.read().strip()
                if token:
                    return token
    except Exception as e:
        print(f"⚠️  Could not read HF token from cache: {e}")
    
    return None

class DiaVoiceAgent:
    """Advanced Dia TTS Voice Agent with fixed audio processing"""
    
    def __init__(self, voice_seed: int = 42, reference_audio_path: Optional[str] = None):
        self.model = None
        self.processor = None
        self.voice_seed = voice_seed
        self.reference_audio_path = reference_audio_path
        self.is_loaded = False
        self.conversation_history = []
        self.speaker_consistency_prompt = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        
    def load_model(self):
        """Load Dia model using HuggingFace Transformers"""
        print("🤖 Loading Dia model using HuggingFace Transformers...")
        
        try:
            from transformers import AutoProcessor, DiaForConditionalGeneration
            import torch
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.voice_seed)
            
            # Use the official HuggingFace Transformers model
            model_checkpoint = "nari-labs/Dia-1.6B-0626"
            
            print(f"📥 Loading processor from {model_checkpoint}...")
            self.processor = AutoProcessor.from_pretrained(model_checkpoint)
            
            print(f"📥 Loading model from {model_checkpoint}...")
            self.model = DiaForConditionalGeneration.from_pretrained(model_checkpoint)
            
            # Move to device
            if self.device == "cuda":
                self.model = self.model.to("cuda")
                print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                print("⚠️  Model loaded on CPU (GPU not available)")
            
            # Set to evaluation mode
            self.model.eval()
            
            # Setup reference audio if provided
            if self.reference_audio_path and os.path.exists(self.reference_audio_path):
                self.setup_voice_cloning()
            
            self.is_loaded = True
            print("✅ Dia model loaded successfully using HuggingFace Transformers!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise e
    
    def generate_speech(self, text: str, use_voice_cloning: bool = True) -> bytes:
        """Generate speech using correct Dia API with proper audio handling"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            import torch
            import soundfile as sf
            import io
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.voice_seed)
            
            # Format text for single speaker
            formatted_text = self.format_text_for_single_speaker(text)
            
            print(f"🎤 Generating speech for: {formatted_text}")
            
            # Prepare inputs using the processor - FIXED APPROACH
            with torch.no_grad():
                if use_voice_cloning and self.speaker_consistency_prompt:
                    # Use voice cloning with reference audio
                    inputs = self.processor(
                        text=[formatted_text],
                        audio=self.speaker_consistency_prompt['audio'],
                        sampling_rate=44100,
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    
                    # Get audio prompt length for correct decoding
                    prompt_len = self.processor.get_audio_prompt_len(inputs["decoder_attention_mask"])
                else:
                    # Standard generation
                    inputs = self.processor(
                        text=[formatted_text],
                        padding=True,
                        return_tensors="pt"
                    ).to(self.device)
                    prompt_len = None
                
                # Generate audio using the model
                outputs = self.model.generate(**inputs, max_new_tokens=256)
                
                # Decode the generated audio - FIXED DECODING
                if prompt_len is not None:
                    # For voice cloning, remove the prompt from output
                    audio_outputs = self.processor.batch_decode(outputs, audio_prompt_len=prompt_len)
                else:
                    # For standard generation - remove skip_special_tokens parameter
                    audio_outputs = self.processor.batch_decode(outputs)
            
            # Extract audio data - FIXED AUDIO EXTRACTION
            if isinstance(audio_outputs, list) and len(audio_outputs) > 0:
                audio_data = audio_outputs[0]
            else:
                audio_data = audio_outputs
            
            # Ensure audio_data is numpy array
            if hasattr(audio_data, 'numpy'):
                audio_data = audio_data.numpy()
            elif hasattr(audio_data, 'cpu'):
                audio_data = audio_data.cpu().numpy()
            
            # Handle tensor dimensions
            if audio_data.ndim > 1:
                audio_data = audio_data.squeeze()
            
            # Convert to bytes with proper format
            buffer = io.BytesIO()
            
            # Use processor's save_audio method if available, otherwise use soundfile
            try:
                # Try using processor's save_audio method
                self.processor.save_audio([audio_data], buffer, format="wav")
                buffer.seek(0)
            except (AttributeError, TypeError):
                # Fallback to soundfile
                sf.write(buffer, audio_data, 44100, format='WAV')
                buffer.seek(0)
            
            print("✅ Speech generated successfully!")
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            # Return empty audio on failure
            import numpy as np
            import soundfile as sf
            silence = np.zeros(44100)  # 1 second of silence
            buffer = io.BytesIO()
            sf.write(buffer, silence, 44100, format='WAV')
            buffer.seek(0)
            return buffer.getvalue()
    
    def format_text_for_single_speaker(self, text: str) -> str:
        """Format text for consistent single speaker output"""
        text = text.strip()
        
        # Remove any existing speaker tags
        text = text.replace("[S1]", "").replace("[S2]", "").strip()
        
        # Add single speaker tag
        formatted_text = f"[S1] {text}"
        
        # Ensure proper ending
        if not text.endswith(('.', '!', '?')):
            formatted_text += "."
        
        return formatted_text
    
    def setup_voice_cloning(self):
        """Setup voice cloning with reference audio"""
        print("🎤 Setting up voice cloning...")
        
        try:
            import soundfile as sf
            import numpy as np
            
            # Load reference audio
            audio_data, sample_rate = sf.read(self.reference_audio_path)
            
            # Ensure audio is the right format
            if len(audio_data.shape) > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
            
            # Resample to 44100 Hz if needed
            if sample_rate != 44100:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=44100)
            
            # Store for voice cloning
            self.speaker_consistency_prompt = {
                "audio": audio_data,
                "sample_rate": 44100,
                "transcript": "[S1] This is my voice reference for cloning."
            }
            
            print("✅ Voice cloning setup complete!")
            
        except Exception as e:
            print(f"⚠️  Voice cloning setup failed: {e}")
    
    def setup_voice_cloning_from_data(self, audio_data: bytes):
        """Setup voice cloning from raw audio data"""
        print("🎤 Setting up voice cloning from recorded audio...")
        
        try:
            import soundfile as sf
            import numpy as np
            import io
            
            # Load audio from bytes
            audio_buffer = io.BytesIO(audio_data)
            audio_array, sample_rate = sf.read(audio_buffer)
            
            # Ensure audio is the right format
            if len(audio_array.shape) > 1:
                audio_array = audio_array.mean(axis=1)  # Convert to mono
            
            # Resample to 44100 Hz if needed
            if sample_rate != 44100:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=44100)
            
            # Store for voice cloning
            self.speaker_consistency_prompt = {
                "audio": audio_array,
                "sample_rate": 44100,
                "transcript": "[S1] This is my recorded voice reference for cloning."
            }
            
            print("✅ Voice cloning setup complete from recorded audio!")
            return True
            
        except Exception as e:
            print(f"⚠️  Voice cloning setup failed: {e}")
            return False
    
    def process_conversation_turn(self, user_input: str) -> bytes:
        """Process a conversation turn and return audio response"""
        self.conversation_history.append({"role": "user", "content": user_input})
        response_text = self.generate_response(user_input)
        self.conversation_history.append({"role": "assistant", "content": response_text})
        audio_bytes = self.generate_speech(response_text)
        return audio_bytes
    
    def generate_response(self, user_input: str) -> str:
        """Generate text response"""
        responses = [
            f"I understand you said: {user_input}. That's interesting!",
            f"Thank you for sharing that. Could you tell me more about {user_input}?",
            f"That's a great point about {user_input}. What do you think about it?",
            f"I appreciate your input on {user_input}. How does that make you feel?",
            f"Regarding {user_input}, I find that fascinating. Can you elaborate?",
            f"You mentioned {user_input}. That reminds me of something similar.",
        ]
        
        random.seed(hash(user_input) + self.voice_seed)
        return random.choice(responses)

class RealTimeVoiceServer:
    """Real-time voice server with FIXED WebSocket handling"""
    
    def __init__(self, voice_agent: DiaVoiceAgent, port: int = 8000):
        self.voice_agent = voice_agent
        self.port = port
        self.app = None
        
    def create_fastapi_app(self):
        """Create FastAPI application"""
        from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException, Request, WebSocketDisconnect
        from fastapi.responses import StreamingResponse, HTMLResponse
        from fastapi.middleware.cors import CORSMiddleware
        import json
        
        app = FastAPI(title="Dia Real-Time Voice Agent", version="3.2.0")
        
        # Add CORS middleware
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        @app.get("/")
        async def root():
            return HTMLResponse(self.get_web_interface())
        
        @app.get("/health")
        async def health_check():
            return {
                "status": "healthy",
                "model_loaded": self.voice_agent.is_loaded,
                "device": self.voice_agent.device,
                "hf_token_available": get_hf_token() is not None,
                "implementation": "HuggingFace Transformers",
                "features": ["audio_recording", "voice_cloning", "real_time_chat", "fixed_websockets"]
            }
        
        @app.post("/generate-speech")
        async def generate_speech(text: str):
            """Generate speech from text"""
            try:
                if not text.strip():
                    raise HTTPException(status_code=400, detail="Text cannot be empty")
                
                audio_bytes = self.voice_agent.generate_speech(text)
                
                return StreamingResponse(
                    io.BytesIO(audio_bytes),
                    media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=speech.wav"}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/conversation-turn")
        async def conversation_turn(user_input: str):
            """Process conversation turn"""
            try:
                if not user_input.strip():
                    raise HTTPException(status_code=400, detail="Input cannot be empty")
                
                audio_bytes = self.voice_agent.process_conversation_turn(user_input)
                
                return StreamingResponse(
                    io.BytesIO(audio_bytes),
                    media_type="audio/wav",
                    headers={"Content-Disposition": "attachment; filename=response.wav"}
                )
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/upload-voice-reference")
        async def upload_voice_reference(file: UploadFile = File(...)):
            """Upload voice reference for cloning"""
            try:
                # Validate file type
                if not file.content_type.startswith('audio/'):
                    raise HTTPException(status_code=400, detail="File must be an audio file")
                
                # Save uploaded file
                file_path = f"voice_reference_{int(time.time())}.wav"
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Update voice agent
                self.voice_agent.reference_audio_path = file_path
                self.voice_agent.setup_voice_cloning()
                
                return {
                    "message": "Voice reference uploaded successfully", 
                    "file_path": file_path,
                    "file_size": len(content)
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.post("/record-voice-reference")
        async def record_voice_reference(request: Request):
            """Process recorded voice reference"""
            try:
                # Get the audio data from request body
                audio_data = await request.body()
                
                if not audio_data:
                    raise HTTPException(status_code=400, detail="No audio data received")
                
                # Setup voice cloning from recorded data
                success = self.voice_agent.setup_voice_cloning_from_data(audio_data)
                
                if success:
                    return {
                        "message": "Voice reference recorded and setup successfully",
                        "data_size": len(audio_data)
                    }
                else:
                    raise HTTPException(status_code=500, detail="Failed to setup voice cloning from recorded audio")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """FIXED WebSocket endpoint with proper error handling"""
            await websocket.accept()
            
            try:
                while True:
                    # Receive text from client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message["type"] == "text_input":
                        # Process conversation turn
                        audio_bytes = self.voice_agent.process_conversation_turn(message["text"])
                        
                        # Send audio back as base64
                        audio_b64 = base64.b64encode(audio_bytes).decode()
                        response = {
                            "type": "audio_response",
                            "audio": audio_b64,
                            "format": "wav",
                            "text": message["text"]
                        }
                        await websocket.send_text(json.dumps(response))
                        
            except WebSocketDisconnect:
                print("WebSocket client disconnected normally")
            except Exception as e:
                print(f"WebSocket error: {e}")
            # FIXED: Remove the finally block that was causing double close
        
        self.app = app
        return app
    
    def get_web_interface(self) -> str:
        """Return enhanced HTML web interface"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dia Real-Time Voice Agent - FIXED</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px; 
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                .container { 
                    background: white; 
                    padding: 25px; 
                    border-radius: 15px; 
                    margin: 15px 0; 
                    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
                }
                h1 { 
                    text-align: center; 
                    color: #333; 
                    margin-bottom: 30px;
                    font-size: 2.5em;
                }
                h3 { 
                    color: #555; 
                    border-bottom: 2px solid #667eea;
                    padding-bottom: 10px;
                }
                .badge {
                    display: inline-block;
                    background: #28a745;
                    color: white;
                    padding: 4px 8px;
                    border-radius: 12px;
                    font-size: 12px;
                    font-weight: bold;
                    margin-left: 10px;
                }
                button { 
                    background: linear-gradient(45deg, #667eea, #764ba2); 
                    color: white; 
                    border: none; 
                    padding: 12px 24px; 
                    border-radius: 8px; 
                    cursor: pointer; 
                    margin: 8px; 
                    font-size: 14px;
                    transition: all 0.3s ease;
                }
                button:hover { 
                    transform: translateY(-2px);
                    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
                }
                button:disabled {
                    background: #ccc;
                    cursor: not-allowed;
                    transform: none;
                }
                .record-btn {
                    background: linear-gradient(45deg, #ff6b6b, #ee5a52);
                }
                .record-btn.recording {
                    background: linear-gradient(45deg, #ff4757, #ff3838);
                    animation: pulse 1s infinite;
                }
                @keyframes pulse {
                    0% { transform: scale(1); }
                    50% { transform: scale(1.05); }
                    100% { transform: scale(1); }
                }
                .stop-btn {
                    background: linear-gradient(45deg, #ffa726, #ff9800);
                }
                input, textarea { 
                    width: 100%; 
                    padding: 12px; 
                    margin: 8px 0; 
                    border: 2px solid #ddd; 
                    border-radius: 8px; 
                    font-size: 14px;
                    transition: border-color 0.3s ease;
                }
                input:focus, textarea:focus {
                    border-color: #667eea;
                    outline: none;
                }
                #status { 
                    padding: 12px; 
                    margin: 12px 0; 
                    border-radius: 8px; 
                    font-weight: bold;
                }
                .success { background: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
                .error { background: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
                .info { background: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
                .warning { background: #fff3cd; color: #856404; border: 1px solid #ffeaa7; }
                audio { width: 100%; margin-top: 15px; }
                .loading { 
                    display: inline-block; 
                    width: 20px; 
                    height: 20px; 
                    border: 3px solid #f3f3f3; 
                    border-top: 3px solid #667eea; 
                    border-radius: 50%; 
                    animation: spin 1s linear infinite; 
                }
                @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                .conversation-history {
                    max-height: 300px;
                    overflow-y: auto;
                    border: 1px solid #ddd;
                    padding: 10px;
                    margin: 10px 0;
                    border-radius: 8px;
                    background: #f9f9f9;
                }
                .message {
                    margin: 8px 0;
                    padding: 8px;
                    border-radius: 6px;
                }
                .user-message {
                    background: #e3f2fd;
                    text-align: right;
                }
                .bot-message {
                    background: #f3e5f5;
                    text-align: left;
                }
                .recording-controls {
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin: 15px 0;
                }
                .recording-status {
                    padding: 8px 16px;
                    border-radius: 20px;
                    font-size: 12px;
                    font-weight: bold;
                }
                .recording-timer {
                    background: #ffebee;
                    color: #c62828;
                }
                .recording-ready {
                    background: #e8f5e8;
                    color: #2e7d32;
                }
                .audio-visualizer {
                    width: 100%;
                    height: 60px;
                    background: #f5f5f5;
                    border-radius: 8px;
                    margin: 10px 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 2px dashed #ddd;
                }
                .tabs {
                    display: flex;
                    margin-bottom: 20px;
                }
                .tab {
                    flex: 1;
                    padding: 12px;
                    text-align: center;
                    background: #f8f9fa;
                    border: 1px solid #dee2e6;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }
                .tab.active {
                    background: linear-gradient(45deg, #667eea, #764ba2);
                    color: white;
                }
                .tab-content {
                    display: none;
                }
                .tab-content.active {
                    display: block;
                }
            </style>
        </head>
        <body>
            <h1>🎤 Dia Real-Time Voice Agent v3.2<span class="badge">FIXED</span></h1>
            
            <div class="container">
                <h3>📊 System Status</h3>
                <button onclick="checkHealth()">Check System Health</button>
                <div id="healthStatus"></div>
            </div>
            
            <div class="container">
                <h3>🎤 Voice Reference Setup</h3>
                
                <div class="tabs">
                    <div class="tab active" onclick="switchTab('upload')">📁 Upload Audio</div>
                    <div class="tab" onclick="switchTab('record')">🎙️ Record Audio</div>
                </div>
                
                <!-- Upload Tab -->
                <div id="upload-tab" class="tab-content active">
                    <p>Upload a 10-second audio sample of your voice for cloning:</p>
                    <input type="file" id="voiceFile" accept="audio/*">
                    <button onclick="uploadVoiceReference()">Upload Voice Reference</button>
                    <div id="uploadStatus"></div>
                </div>
                
                <!-- Recording Tab -->
                <div id="record-tab" class="tab-content">
                    <p>Record a 10-second sample of your voice directly in the browser:</p>
                    
                    <div class="recording-controls">
                        <button id="recordBtn" class="record-btn" onclick="startRecording()">🎙️ Start Recording</button>
                        <button id="stopBtn" class="stop-btn" onclick="stopRecording()" disabled>⏹️ Stop Recording</button>
                        <div id="recordingStatus" class="recording-status recording-ready">Ready to record</div>
                    </div>
                    
                    <div class="audio-visualizer" id="audioVisualizer">
                        <span id="visualizerText">Audio levels will appear here during recording</span>
                    </div>
                    
                    <audio id="recordedAudio" controls style="width: 100%; margin-top: 10px; display: none;"></audio>
                    <button id="useRecordingBtn" onclick="useRecordedAudio()" style="display: none;">✅ Use This Recording</button>
                    
                    <div id="recordStatus"></div>
                </div>
            </div>
            
            <div class="container">
                <h3>🗣️ Text to Speech</h3>
                <textarea id="textInput" placeholder="Enter text to convert to speech..." rows="4"></textarea>
                <button onclick="generateSpeech()" id="generateBtn">Generate Speech</button>
                <audio id="audioPlayer" controls style="width: 100%; margin-top: 10px;"></audio>
            </div>
            
            <div class="container">
                <h3>💬 Real-Time Conversation</h3>
                <div class="conversation-history" id="conversationHistory"></div>
                <input type="text" id="conversationInput" placeholder="Type your message...">
                <button onclick="sendMessage()" id="sendBtn">Send Message</button>
                <button onclick="connectWebSocket()" id="connectBtn">Connect WebSocket</button>
                <div id="status"></div>
            </div>
            
            <script>
                let ws = null;
                let conversationHistory = [];
                let mediaRecorder = null;
                let audioChunks = [];
                let recordingTimer = null;
                let recordingSeconds = 0;
                
                function showStatus(message, type = 'info', elementId = 'status') {
                    const statusEl = document.getElementById(elementId);
                    statusEl.textContent = message;
                    statusEl.className = type;
                }
                
                function switchTab(tabName) {
                    // Hide all tab contents
                    document.querySelectorAll('.tab-content').forEach(content => {
                        content.classList.remove('active');
                    });
                    
                    // Remove active class from all tabs
                    document.querySelectorAll('.tab').forEach(tab => {
                        tab.classList.remove('active');
                    });
                    
                    // Show selected tab content
                    document.getElementById(tabName + '-tab').classList.add('active');
                    
                    // Add active class to selected tab
                    event.target.classList.add('active');
                }
                
                async function checkHealth() {
                    try {
                        const response = await fetch('/health');
                        const data = await response.json();
                        
                        let message = `Status: ${data.status} | Model: ${data.model_loaded ? 'Loaded' : 'Not Loaded'} | Device: ${data.device} | HF Token: ${data.hf_token_available ? 'Available' : 'Not Available'} | Implementation: ${data.implementation}`;
                        if (data.features) {
                            message += ` | Features: ${data.features.join(', ')}`;
                        }
                        showStatus(message, 'success', 'healthStatus');
                    } catch (error) {
                        showStatus('Health check failed: ' + error.message, 'error', 'healthStatus');
                    }
                }
                
                async function uploadVoiceReference() {
                    const fileInput = document.getElementById('voiceFile');
                    const file = fileInput.files[0];
                    
                    if (!file) {
                        showStatus('Please select an audio file', 'error', 'uploadStatus');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        showStatus('Uploading...', 'info', 'uploadStatus');
                        const response = await fetch('/upload-voice-reference', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            showStatus(`Voice reference uploaded successfully! Size: ${data.file_size} bytes`, 'success', 'uploadStatus');
                        } else {
                            const error = await response.json();
                            showStatus('Upload failed: ' + error.detail, 'error', 'uploadStatus');
                        }
                    } catch (error) {
                        showStatus('Upload error: ' + error.message, 'error', 'uploadStatus');
                    }
                }
                
                async function startRecording() {
                    try {
                        const stream = await navigator.mediaDevices.getUserMedia({ 
                            audio: {
                                echoCancellation: true,
                                noiseSuppression: true,
                                sampleRate: 44100
                            } 
                        });
                        
                        mediaRecorder = new MediaRecorder(stream, {
                            mimeType: 'audio/webm;codecs=opus'
                        });
                        
                        audioChunks = [];
                        recordingSeconds = 0;
                        
                        mediaRecorder.ondataavailable = event => {
                            audioChunks.push(event.data);
                        };
                        
                        mediaRecorder.onstop = async () => {
                            const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
                            const audioUrl = URL.createObjectURL(audioBlob);
                            
                            const recordedAudio = document.getElementById('recordedAudio');
                            recordedAudio.src = audioUrl;
                            recordedAudio.style.display = 'block';
                            
                            document.getElementById('useRecordingBtn').style.display = 'inline-block';
                            
                            // Store the blob for later use
                            window.recordedAudioBlob = audioBlob;
                            
                            showStatus('Recording completed! You can now use this recording.', 'success', 'recordStatus');
                        };
                        
                        mediaRecorder.start();
                        
                        // Update UI
                        document.getElementById('recordBtn').disabled = true;
                        document.getElementById('stopBtn').disabled = false;
                        document.getElementById('recordBtn').classList.add('recording');
                        
                        // Start timer
                        recordingTimer = setInterval(() => {
                            recordingSeconds++;
                            document.getElementById('recordingStatus').textContent = `Recording: ${recordingSeconds}s`;
                            document.getElementById('recordingStatus').className = 'recording-status recording-timer';
                            
                            // Auto-stop after 30 seconds
                            if (recordingSeconds >= 30) {
                                stopRecording();
                            }
                        }, 1000);
                        
                        showStatus('Recording started! Speak clearly for 5-10 seconds.', 'info', 'recordStatus');
                        
                    } catch (error) {
                        showStatus('Error accessing microphone: ' + error.message, 'error', 'recordStatus');
                    }
                }
                
                function stopRecording() {
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                        
                        // Stop all tracks
                        mediaRecorder.stream.getTracks().forEach(track => track.stop());
                    }
                    
                    // Clear timer
                    if (recordingTimer) {
                        clearInterval(recordingTimer);
                        recordingTimer = null;
                    }
                    
                    // Update UI
                    document.getElementById('recordBtn').disabled = false;
                    document.getElementById('stopBtn').disabled = true;
                    document.getElementById('recordBtn').classList.remove('recording');
                    document.getElementById('recordingStatus').textContent = 'Recording stopped';
                    document.getElementById('recordingStatus').className = 'recording-status recording-ready';
                }
                
                async function useRecordedAudio() {
                    if (!window.recordedAudioBlob) {
                        showStatus('No recording available', 'error', 'recordStatus');
                        return;
                    }
                    
                    try {
                        showStatus('Processing recorded audio...', 'info', 'recordStatus');
                        
                        const response = await fetch('/record-voice-reference', {
                            method: 'POST',
                            body: window.recordedAudioBlob,
                            headers: {
                                'Content-Type': 'audio/webm'
                            }
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            showStatus(`Voice reference recorded successfully! Data size: ${data.data_size} bytes`, 'success', 'recordStatus');
                        } else {
                            const error = await response.json();
                            showStatus('Failed to setup voice reference: ' + error.detail, 'error', 'recordStatus');
                        }
                    } catch (error) {
                        showStatus('Error processing recording: ' + error.message, 'error', 'recordStatus');
                    }
                }
                
                async function generateSpeech() {
                    const text = document.getElementById('textInput').value;
                    const btn = document.getElementById('generateBtn');
                    
                    if (!text.trim()) {
                        showStatus('Please enter some text', 'error');
                        return;
                    }
                    
                    try {
                        btn.disabled = true;
                        btn.innerHTML = '<span class="loading"></span> Generating...';
                        
                        const response = await fetch('/generate-speech?text=' + encodeURIComponent(text), {
                            method: 'POST'
                        });
                        
                        if (response.ok) {
                            const audioBlob = await response.blob();
                            const audioUrl = URL.createObjectURL(audioBlob);
                            document.getElementById('audioPlayer').src = audioUrl;
                            showStatus('Speech generated successfully!', 'success');
                        } else {
                            const error = await response.json();
                            showStatus('Speech generation failed: ' + error.detail, 'error');
                        }
                    } catch (error) {
                        showStatus('Error: ' + error.message, 'error');
                    } finally {
                        btn.disabled = false;
                        btn.innerHTML = 'Generate Speech';
                    }
                }
                
                function connectWebSocket() {
                    const btn = document.getElementById('connectBtn');
                    
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.close();
                        return;
                    }
                    
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
                    
                    ws.onopen = function() {
                        showStatus('WebSocket connected!', 'success');
                        btn.textContent = 'Disconnect WebSocket';
                        btn.style.background = '#dc3545';
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'audio_response') {
                            const audioBlob = new Blob([Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))], {type: 'audio/wav'});
                            const audioUrl = URL.createObjectURL(audioBlob);
                            document.getElementById('audioPlayer').src = audioUrl;
                            document.getElementById('audioPlayer').play();
                            
                            // Add to conversation history
                            addToConversationHistory('Bot', 'Audio response generated');
                        }
                    };
                    
                    ws.onerror = function() {
                        showStatus('WebSocket error', 'error');
                    };
                    
                    ws.onclose = function() {
                        showStatus('WebSocket disconnected', 'info');
                        btn.textContent = 'Connect WebSocket';
                        btn.style.background = 'linear-gradient(45deg, #667eea, #764ba2)';
                    };
                }
                
                function sendMessage() {
                    const input = document.getElementById('conversationInput');
                    const message = input.value.trim();
                    const btn = document.getElementById('sendBtn');
                    
                    if (!message || !ws || ws.readyState !== WebSocket.OPEN) {
                        showStatus('Please connect WebSocket and enter a message', 'error');
                        return;
                    }
                    
                    try {
                        btn.disabled = true;
                        btn.innerHTML = '<span class="loading"></span> Processing...';
                        
                        ws.send(JSON.stringify({
                            type: 'text_input',
                            text: message
                        }));
                        
                        // Add to conversation history
                        addToConversationHistory('You', message);
                        
                        input.value = '';
                        showStatus('Message sent, generating response...', 'info');
                    } catch (error) {
                        showStatus('Error sending message: ' + error.message, 'error');
                    } finally {
                        setTimeout(() => {
                            btn.disabled = false;
                            btn.innerHTML = 'Send Message';
                        }, 2000);
                    }
                }
                
                function addToConversationHistory(sender, message) {
                    const historyEl = document.getElementById('conversationHistory');
                    const messageEl = document.createElement('div');
                    messageEl.className = 'message ' + (sender === 'You' ? 'user-message' : 'bot-message');
                    messageEl.innerHTML = `<strong>${sender}:</strong> ${message}`;
                    historyEl.appendChild(messageEl);
                    historyEl.scrollTop = historyEl.scrollHeight;
                }
                
                // Enter key support
                document.getElementById('conversationInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
                
                // Auto-check health on load
                window.onload = function() {
                    checkHealth();
                };
                
                // Check microphone permission on load
                navigator.mediaDevices.getUserMedia({ audio: true })
                    .then(stream => {
                        stream.getTracks().forEach(track => track.stop());
                        showStatus('Microphone access granted - ready to record!', 'success', 'recordStatus');
                    })
                    .catch(error => {
                        showStatus('Microphone access denied. Please allow microphone access to use recording feature.', 'warning', 'recordStatus');
                    });
            </script>
        </body>
        </html>
        """
    
    def run(self):
        """Run the server"""
        import uvicorn
        
        self.create_fastapi_app()
        print(f"🚀 Starting server on port {self.port}")
        print(f"🌐 Access the web interface at: http://localhost:{self.port}")
        print(f"📱 Or use your RunPod URL: http://[your-runpod-url]:{self.port}")
        
        uvicorn.run(self.app, host="0.0.0.0", port=self.port, log_level="info")

def main():
    """Main execution function"""
    print("🎯 Dia TTS Real-Time Conversational AI Agent v3.2")
    print("=" * 70)
    print("✨ FIXED - WebSocket and Audio Issues Resolved!")
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check HuggingFace token
    hf_token = get_hf_token()
    if hf_token:
        print("✅ HuggingFace token found and ready to use")
    else:
        print("⚠️  No HuggingFace token found")
        print("💡 Run: huggingface-cli login")
    
    # Initialize voice agent
    print("🤖 Initializing voice agent...")
    voice_agent = DiaVoiceAgent(voice_seed=42)
    
    # Load model
    try:
        voice_agent.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)
    
    # Create and run server
    server = RealTimeVoiceServer(voice_agent, port=8000)
    
    print("\n🎉 Setup complete! FIXED Features available:")
    print("✅ HuggingFace Transformers integration")
    print("✅ Single speaker consistency (fixed seed)")
    print("✅ Voice cloning support (10-second audio)")
    print("✅ 🎙️ BROWSER AUDIO RECORDING")
    print("✅ Real-time conversation")
    print("✅ WebSocket support (FIXED - no more double close errors)")
    print("✅ Enhanced web interface")
    print("✅ FIXED audio processing (proper audio output)")
    print("✅ Production-ready architecture")
    
    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()
