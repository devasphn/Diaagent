
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

class DiaVoiceAgent:
    """Advanced Dia TTS Voice Agent with real-time capabilities"""
    
    def __init__(self, voice_seed: int = 42, reference_audio_path: Optional[str] = None):
        self.model = None
        self.voice_seed = voice_seed
        self.reference_audio_path = reference_audio_path
        self.is_loaded = False
        self.conversation_history = []
        self.speaker_consistency_prompt = None
        
    def load_model(self):
        """Load Dia model with correct parameters"""
        print("🤖 Loading Dia model...")
        
        try:
            import torch
            from dia.model import Dia
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            torch.cuda.manual_seed_all(self.voice_seed)
            random.seed(self.voice_seed)
            
            # Load model with correct parameters (no device_map or compute_dtype)
            self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
            
            # Move model to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                print("✅ Model loaded on GPU")
            else:
                print("⚠️  Model loaded on CPU")
            
            # Setup reference audio if provided
            if self.reference_audio_path and os.path.exists(self.reference_audio_path):
                self.setup_voice_cloning()
            
            self.is_loaded = True
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def setup_voice_cloning(self):
        """Setup voice cloning with reference audio"""
        print("🎤 Setting up voice cloning...")
        
        try:
            import soundfile as sf
            import numpy as np
            
            # Load reference audio
            audio_data, sample_rate = sf.read(self.reference_audio_path)
            
            # Resample to 44100 Hz if needed
            if sample_rate != 44100:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=44100)
            
            # Ensure audio is mono
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Create consistency prompt
            self.speaker_consistency_prompt = {
                "audio": audio_data,
                "sample_rate": 44100,
                "transcript": "[S1] This is my voice reference for cloning."
            }
            
            print("✅ Voice cloning setup complete!")
            
        except Exception as e:
            print(f"⚠️  Voice cloning setup failed: {e}")
    
    def generate_speech(self, text: str, use_voice_cloning: bool = True) -> bytes:
        """Generate speech from text with consistency"""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        try:
            import torch
            import soundfile as sf
            import numpy as np
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            torch.cuda.manual_seed_all(self.voice_seed)
            random.seed(self.voice_seed)
            
            # Format text for single speaker
            formatted_text = self.format_text_for_single_speaker(text)
            
            # Generate audio using the correct API
            if use_voice_cloning and self.speaker_consistency_prompt:
                # Use voice cloning with reference audio
                print("🎤 Generating with voice cloning...")
                output = self.model.generate(
                    formatted_text,
                    reference_audio=self.speaker_consistency_prompt['audio'],
                    seed=self.voice_seed
                )
            else:
                # Standard generation with fixed seed
                print("🗣️  Generating with fixed seed...")
                output = self.model.generate(
                    formatted_text,
                    seed=self.voice_seed
                )
            
            # Ensure output is numpy array
            if hasattr(output, 'cpu'):
                output = output.cpu().numpy()
            
            # Convert to bytes
            buffer = io.BytesIO()
            sf.write(buffer, output, 44100, format='WAV')
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            # Return empty audio on failure
            import soundfile as sf
            silence = np.zeros(44100)  # 1 second of silence
            buffer = io.BytesIO()
            sf.write(buffer, silence, 44100, format='WAV')
            return buffer.getvalue()
    
    def format_text_for_single_speaker(self, text: str) -> str:
        """Format text for consistent single speaker output"""
        # Clean and format text
        text = text.strip()
        
        # Remove existing speaker tags to ensure single speaker
        text = text.replace("[S1]", "").replace("[S2]", "").strip()
        
        # Add single speaker tag
        text = f"[S1] {text}"
        
        return text
    
    def process_conversation_turn(self, user_input: str) -> bytes:
        """Process a conversation turn and return audio response"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response (you can integrate with LLM here)
        response_text = self.generate_response(user_input)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Generate speech
        audio_bytes = self.generate_speech(response_text)
        
        return audio_bytes
    
    def generate_response(self, user_input: str) -> str:
        """Generate text response (integrate with your preferred LLM)"""
        # Simple responses for demonstration
        responses = [
            f"I understand you said: {user_input}. That's interesting!",
            f"Thank you for sharing that. Could you tell me more about {user_input}?",
            f"That's a great point about {user_input}. What do you think about it?",
            f"I appreciate your input on {user_input}. How does that make you feel?",
            f"Regarding {user_input}, I find that fascinating. Can you elaborate?",
            f"Your comment about {user_input} is thought-provoking. What's your perspective?",
        ]
        
        # Use seed for consistent but varied responses
        random.seed(self.voice_seed + len(self.conversation_history))
        return random.choice(responses)

class RealTimeVoiceServer:
    """Real-time voice server with WebSocket support"""
    
    def __init__(self, voice_agent: DiaVoiceAgent, port: int = 8000):
        self.voice_agent = voice_agent
        self.port = port
        self.app = None
        
    def create_fastapi_app(self):
        """Create FastAPI application"""
        from fastapi import FastAPI, WebSocket, UploadFile, File, HTTPException
        from fastapi.responses import StreamingResponse, HTMLResponse
        from fastapi.middleware.cors import CORSMiddleware
        import json
        
        app = FastAPI(title="Dia Real-Time Voice Agent", version="1.0.0")
        
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
        async def health():
            return {"status": "healthy", "model_loaded": self.voice_agent.is_loaded}
        
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
                if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a')):
                    raise HTTPException(status_code=400, detail="Unsupported audio format")
                
                # Save uploaded file
                file_path = f"voice_reference_{int(time.time())}.wav"
                with open(file_path, "wb") as f:
                    content = await file.read()
                    f.write(content)
                
                # Update voice agent
                self.voice_agent.reference_audio_path = file_path
                self.voice_agent.setup_voice_cloning()
                
                return {"message": "Voice reference uploaded successfully", "file_path": file_path}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time communication"""
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
                            "format": "wav"
                        }
                        await websocket.send_text(json.dumps(response))
                        
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                await websocket.close()
        
        self.app = app
        return app
    
    def get_web_interface(self) -> str:
        """Return HTML web interface"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dia Real-Time Voice Agent</title>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 800px; 
                    margin: 0 auto; 
                    padding: 20px; 
                    background: #f0f2f5;
                }
                .container { 
                    background: white; 
                    padding: 20px; 
                    border-radius: 10px; 
                    margin: 10px 0; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                button { 
                    background: #007bff; 
                    color: white; 
                    border: none; 
                    padding: 10px 20px; 
                    border-radius: 5px; 
                    cursor: pointer; 
                    margin: 5px;
                    transition: background 0.3s;
                }
                button:hover { background: #0056b3; }
                button:disabled { background: #ccc; cursor: not-allowed; }
                input, textarea { 
                    width: 100%; 
                    padding: 10px; 
                    margin: 5px 0; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    box-sizing: border-box;
                }
                #status { 
                    padding: 10px; 
                    margin: 10px 0; 
                    border-radius: 5px; 
                    font-weight: bold;
                }
                .success { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
                .loading { background: #fff3cd; color: #856404; }
                .header { 
                    text-align: center; 
                    color: #333; 
                    margin-bottom: 20px;
                }
                .feature-list {
                    list-style: none;
                    padding: 0;
                }
                .feature-list li {
                    padding: 5px 0;
                    border-bottom: 1px solid #eee;
                }
                .feature-list li:before {
                    content: "✅ ";
                    color: #28a745;
                }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🤖 Dia Real-Time Voice Agent</h1>
                <p>Ultra-realistic AI voice generation with conversation capabilities</p>
            </div>
            
            <div class="container">
                <h3>📊 System Status</h3>
                <button onclick="checkHealth()">Check System Health</button>
                <div id="healthStatus"></div>
            </div>
            
            <div class="container">
                <h3>🎤 Voice Reference Upload</h3>
                <p>Upload a 10-second audio sample for voice cloning:</p>
                <input type="file" id="voiceFile" accept="audio/*">
                <button onclick="uploadVoiceReference()">Upload Voice Reference</button>
            </div>
            
            <div class="container">
                <h3>🗣️ Text to Speech</h3>
                <textarea id="textInput" placeholder="Enter text to convert to speech..." rows="3"></textarea>
                <button onclick="generateSpeech()">Generate Speech</button>
                <audio id="audioPlayer" controls style="width: 100%; margin-top: 10px;"></audio>
            </div>
            
            <div class="container">
                <h3>💬 Real-Time Conversation</h3>
                <input type="text" id="conversationInput" placeholder="Type your message...">
                <button onclick="sendMessage()">Send Message</button>
                <button onclick="connectWebSocket()">Connect WebSocket</button>
                <div id="status"></div>
            </div>
            
            <div class="container">
                <h3>✨ Features</h3>
                <ul class="feature-list">
                    <li>Single speaker consistency with fixed seed</li>
                    <li>Voice cloning with 10-second audio samples</li>
                    <li>Real-time conversation processing</li>
                    <li>WebSocket support for instant responses</li>
                    <li>Professional web interface</li>
                    <li>RESTful API endpoints</li>
                </ul>
            </div>
            
            <script>
                let ws = null;
                let isGenerating = false;
                
                function showStatus(message, type = 'success') {
                    const status = document.getElementById('status');
                    status.textContent = message;
                    status.className = type;
                }
                
                function showHealthStatus(message, type = 'success') {
                    const status = document.getElementById('healthStatus');
                    status.textContent = message;
                    status.className = type;
                }
                
                async function checkHealth() {
                    try {
                        const response = await fetch('/health');
                        const data = await response.json();
                        
                        if (data.model_loaded) {
                            showHealthStatus('✅ System healthy - Model loaded and ready', 'success');
                        } else {
                            showHealthStatus('⚠️ System healthy - Model not loaded', 'loading');
                        }
                    } catch (error) {
                        showHealthStatus('❌ System error: ' + error.message, 'error');
                    }
                }
                
                async function uploadVoiceReference() {
                    const fileInput = document.getElementById('voiceFile');
                    const file = fileInput.files[0];
                    
                    if (!file) {
                        showStatus('Please select a file', 'error');
                        return;
                    }
                    
                    if (file.size > 50 * 1024 * 1024) { // 50MB limit
                        showStatus('File too large. Please use a file smaller than 50MB.', 'error');
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        showStatus('Uploading voice reference...', 'loading');
                        const response = await fetch('/upload-voice-reference', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            showStatus('Voice reference uploaded successfully!', 'success');
                        } else {
                            const error = await response.json();
                            showStatus('Upload failed: ' + error.detail, 'error');
                        }
                    } catch (error) {
                        showStatus('Upload error: ' + error.message, 'error');
                    }
                }
                
                async function generateSpeech() {
                    const text = document.getElementById('textInput').value;
                    
                    if (!text.trim()) {
                        showStatus('Please enter some text', 'error');
                        return;
                    }
                    
                    if (isGenerating) {
                        showStatus('Already generating speech, please wait...', 'loading');
                        return;
                    }
                    
                    try {
                        isGenerating = true;
                        showStatus('Generating speech...', 'loading');
                        
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
                        isGenerating = false;
                    }
                }
                
                function connectWebSocket() {
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        showStatus('WebSocket already connected!', 'success');
                        return;
                    }
                    
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
                    
                    ws.onopen = function() {
                        showStatus('WebSocket connected!', 'success');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'audio_response') {
                            const audioBlob = new Blob([Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))], {type: 'audio/wav'});
                            const audioUrl = URL.createObjectURL(audioBlob);
                            document.getElementById('audioPlayer').src = audioUrl;
                            document.getElementById('audioPlayer').play();
                            showStatus('Response received and playing!', 'success');
                        }
                    };
                    
                    ws.onerror = function() {
                        showStatus('WebSocket error occurred', 'error');
                    };
                    
                    ws.onclose = function() {
                        showStatus('WebSocket disconnected', 'error');
                    };
                }
                
                function sendMessage() {
                    const input = document.getElementById('conversationInput');
                    const message = input.value.trim();
                    
                    if (!message) {
                        showStatus('Please enter a message', 'error');
                        return;
                    }
                    
                    if (!ws || ws.readyState !== WebSocket.OPEN) {
                        showStatus('Please connect WebSocket first', 'error');
                        return;
                    }
                    
                    ws.send(JSON.stringify({
                        type: 'text_input',
                        text: message
                    }));
                    
                    input.value = '';
                    showStatus('Message sent, generating response...', 'loading');
                }
                
                // Enter key support
                document.getElementById('conversationInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
                });
                
                // Auto-check health on page load
                window.addEventListener('load', checkHealth);
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
        print(f"📡 WebSocket endpoint: ws://localhost:{self.port}/ws")
        print(f"🔗 API documentation: http://localhost:{self.port}/docs")
        
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

def main():
    """Main execution function"""
    print("🎯 Dia TTS Real-Time Conversational AI Agent - Fixed Version")
    print("=" * 60)
    
    # Check if model is already installed
    try:
        import dia
        print("✅ Dia library found!")
    except ImportError:
        print("❌ Dia library not found. Please install it first:")
        print("pip install -e .")
        sys.exit(1)
    
    # Initialize voice agent
    print("🤖 Initializing voice agent...")
    voice_agent = DiaVoiceAgent(
        voice_seed=42,  # Fixed seed for consistency
        reference_audio_path=None  # Set this to your voice reference file
    )
    
    # Load model
    try:
        voice_agent.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("Please check your GPU memory and CUDA installation.")
        sys.exit(1)
    
    # Create and run server
    server = RealTimeVoiceServer(voice_agent, port=8000)
    
    print("\n🎉 Setup complete! Features available:")
    print("✅ Single speaker consistency (fixed seed)")
    print("✅ Voice cloning support (10-second audio)")
    print("✅ Real-time conversation")
    print("✅ WebSocket support")
    print("✅ Professional web interface")
    print("✅ REST API endpoints")
    print("✅ Health monitoring")
    
    try:
        # Run server
        server.run()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")

if __name__ == "__main__":
    main()
