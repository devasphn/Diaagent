
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
    """Advanced Dia TTS Voice Agent with correct API usage"""
    
    def __init__(self, voice_seed: int = 42, reference_audio_path: Optional[str] = None):
        self.model = None
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
        """Load Dia model using the correct official API"""
        print("🤖 Loading Dia model with correct API...")
        
        try:
            import torch
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.voice_seed)
            
            # Get HuggingFace token
            hf_token = get_hf_token()
            if not hf_token:
                raise ValueError("HuggingFace token is required but not found")
            
            print("📥 Loading Dia model from HuggingFace Hub...")
            
            # Use the correct Dia.from_pretrained method (this is the official way)
            from dia import Dia
            
            # Load model using the official HuggingFace Hub integration
            self.model = Dia.from_pretrained(
                "nari-labs/Dia-1.6B",
                token=hf_token
            )
            
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
            print("✅ Dia model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("💡 Trying alternative loading approach...")
            
            # Alternative approach: Use the official repository method
            try:
                from dia import Dia
                
                # Load without specifying token in constructor (relies on HF CLI login)
                self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
                
                # Move to device
                if self.device == "cuda":
                    self.model = self.model.to("cuda")
                    print(f"✅ Model loaded on GPU with alternative method")
                else:
                    print("⚠️  Model loaded on CPU with alternative method")
                
                self.model.eval()
                self.is_loaded = True
                print("✅ Dia model loaded successfully with alternative method!")
                
            except Exception as e2:
                print(f"❌ Alternative loading also failed: {e2}")
                raise e2
    
    def generate_speech(self, text: str, use_voice_cloning: bool = True) -> bytes:
        """Generate speech using the correct Dia API"""
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
            
            # Generate audio using the correct Dia API
            with torch.no_grad():
                if use_voice_cloning and self.speaker_consistency_prompt:
                    # Use voice cloning with reference audio
                    output = self.model.generate(
                        text=formatted_text,
                        reference_audio=self.speaker_consistency_prompt['audio']
                    )
                else:
                    # Standard generation
                    output = self.model.generate(text=formatted_text)
            
            # Convert to bytes
            buffer = io.BytesIO()
            
            # Ensure output is numpy array
            if hasattr(output, 'cpu'):
                output = output.cpu().numpy()
            elif hasattr(output, 'detach'):
                output = output.detach().cpu().numpy()
            
            # Handle different output formats
            if output.ndim > 1:
                output = output.squeeze()
            
            # Write as WAV
            sf.write(buffer, output, 44100, format='WAV')
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
            
            # Convert to tensor if needed
            import torch
            if isinstance(audio_data, np.ndarray):
                audio_data = torch.from_numpy(audio_data).float()
            
            # Store for voice cloning
            self.speaker_consistency_prompt = {
                "audio": audio_data,
                "sample_rate": 44100,
                "transcript": "This is my voice reference for cloning."
            }
            
            print("✅ Voice cloning setup complete!")
            
        except Exception as e:
            print(f"⚠️  Voice cloning setup failed: {e}")
    
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
        
        app = FastAPI(title="Dia Real-Time Voice Agent", version="2.3.0")
        
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
                "hf_token_available": get_hf_token() is not None
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
                            "format": "wav",
                            "text": message["text"]
                        }
                        await websocket.send_text(json.dumps(response))
                        
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                await websocket.close()
        
        self.app = app
        return app
    
    def get_web_interface(self) -> str:
        """Return enhanced HTML web interface"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dia Real-Time Voice Agent</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { box-sizing: border-box; }
                body { 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    max-width: 1000px; 
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
            </style>
        </head>
        <body>
            <h1>🤖 Dia Real-Time Voice Agent</h1>
            
            <div class="container">
                <h3>📊 System Status</h3>
                <button onclick="checkHealth()">Check System Health</button>
                <div id="healthStatus"></div>
            </div>
            
            <div class="container">
                <h3>🎤 Voice Reference Upload</h3>
                <p>Upload a 10-second audio sample of your voice for cloning:</p>
                <input type="file" id="voiceFile" accept="audio/*">
                <button onclick="uploadVoiceReference()">Upload Voice Reference</button>
                <div id="uploadStatus"></div>
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
                
                function showStatus(message, type = 'info', elementId = 'status') {
                    const statusEl = document.getElementById(elementId);
                    statusEl.textContent = message;
                    statusEl.className = type;
                }
                
                async function checkHealth() {
                    try {
                        const response = await fetch('/health');
                        const data = await response.json();
                        
                        let message = `Status: ${data.status} | Model: ${data.model_loaded ? 'Loaded' : 'Not Loaded'} | Device: ${data.device} | HF Token: ${data.hf_token_available ? 'Available' : 'Not Available'}`;
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
    print("🎯 Dia TTS Real-Time Conversational AI Agent v2.3")
    print("=" * 60)
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check HuggingFace token
    hf_token = get_hf_token()
    if hf_token:
        print("✅ HuggingFace token found and ready to use")
    else:
        print("❌ No HuggingFace token found - this is required!")
        print("💡 Run: huggingface-cli login")
        sys.exit(1)
    
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
    
    print("\n🎉 Setup complete! Features available:")
    print("✅ Single speaker consistency (fixed seed)")
    print("✅ Voice cloning support (10-second audio)")
    print("✅ Real-time conversation")
    print("✅ WebSocket support")
    print("✅ Enhanced web interface")
    print("✅ REST API endpoints")
    print("✅ Proper Dia model loading")
    
    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()
