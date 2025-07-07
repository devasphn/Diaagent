
import os
import sys
import subprocess
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

# Suppress warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

class DiaSetup:
    """Handles complete Dia TTS setup and installation"""
    
    def __init__(self):
        self.setup_complete = False
        
    def install_system_dependencies(self):
        """Install system-level dependencies"""
        print("🔧 Installing system dependencies...")
        
        commands = [
            "apt-get update -qq",
            "apt-get install -y git wget curl ffmpeg sox libsox-fmt-all",
            "pip install --upgrade pip setuptools wheel",
        ]
        
        for cmd in commands:
            subprocess.run(cmd.split(), check=True, capture_output=True)
    
    def install_python_dependencies(self):
        """Install Python dependencies"""
        print("📦 Installing Python dependencies...")
        
        # Core dependencies
        packages = [
            "torch>=2.0.0",
            "torchaudio",
            "transformers>=4.30.0",
            "accelerate",
            "gradio",
            "soundfile",
            "librosa",
            "numpy",
            "scipy",
            "fastapi",
            "uvicorn",
            "websockets",
            "aiofiles",
            "python-multipart",
            "pydantic",
            "requests",
            "huggingface_hub",
            "datasets",
            "tokenizers",
        ]
        
        for package in packages:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         check=True, capture_output=True)
    
    def clone_and_setup_dia(self):
        """Clone Dia repository and install"""
        print("🚀 Setting up Dia TTS...")
        
        if not os.path.exists("dia"):
            subprocess.run(["git", "clone", "https://github.com/nari-labs/dia.git"], 
                         check=True)
        
        os.chdir("dia")
        subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], 
                     check=True)
        os.chdir("..")
    
    def setup_environment(self):
        """Setup environment variables"""
        print("🔑 Setting up environment...")
        
        # Set HuggingFace token if available
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            print("⚠️  HF_TOKEN not set. You may need to set it for model access.")
        
        # Set CUDA environment
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    def complete_setup(self):
        """Run complete setup process"""
        if self.setup_complete:
            return
            
        try:
            self.install_system_dependencies()
            self.install_python_dependencies()
            self.clone_and_setup_dia()
            self.setup_environment()
            self.setup_complete = True
            print("✅ Setup completed successfully!")
        except Exception as e:
            print(f"❌ Setup failed: {e}")
            raise

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
        """Load Dia model with optimizations"""
        print("🤖 Loading Dia model...")
        
        try:
            from dia.model import Dia
            import torch
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            torch.cuda.manual_seed(self.voice_seed)
            
            # Load model with optimizations
            self.model = Dia.from_pretrained(
                "nari-labs/Dia-1.6B",
                compute_dtype="float16",  # Use float16 for A40 efficiency
                device_map="auto"
            )
            
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
            
            # Load reference audio
            audio_data, sample_rate = sf.read(self.reference_audio_path)
            
            # Create consistency prompt
            self.speaker_consistency_prompt = {
                "audio": audio_data,
                "sample_rate": sample_rate,
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
            import io
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            torch.cuda.manual_seed(self.voice_seed)
            
            # Format text for single speaker
            formatted_text = self.format_text_for_single_speaker(text)
            
            # Generate audio
            if use_voice_cloning and self.speaker_consistency_prompt:
                # Use voice cloning
                full_text = f"{self.speaker_consistency_prompt['transcript']} {formatted_text}"
                output = self.model.generate(
                    full_text,
                    audio_prompt=self.speaker_consistency_prompt['audio'],
                    use_torch_compile=True,
                    verbose=False,
                    temperature=0.7,
                    cfg_scale=3.0
                )
            else:
                # Standard generation with fixed seed
                output = self.model.generate(
                    formatted_text,
                    use_torch_compile=True,
                    verbose=False,
                    temperature=0.7,
                    cfg_scale=3.0
                )
            
            # Convert to bytes
            buffer = io.BytesIO()
            sf.write(buffer, output, 44100, format='WAV')
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            raise
    
    def format_text_for_single_speaker(self, text: str) -> str:
        """Format text for consistent single speaker output"""
        # Clean and format text
        text = text.strip()
        
        # Ensure it starts with [S1]
        if not text.startswith("[S1]"):
            text = f"[S1] {text}"
        
        # End with speaker tag for better quality
        if not text.endswith("[S1]"):
            text = f"{text} [S1]"
        
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
        # Simple echo response for demonstration
        # Replace this with your LLM integration
        responses = [
            f"I understand you said: {user_input}. That's interesting!",
            f"Thank you for sharing that. Could you tell me more about {user_input}?",
            f"That's a great point about {user_input}. What do you think about it?",
            f"I appreciate your input on {user_input}. How does that make you feel?",
        ]
        
        import random
        random.seed(self.voice_seed)  # Consistent responses
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
        
        @app.post("/generate-speech")
        async def generate_speech(text: str):
            """Generate speech from text"""
            try:
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
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                .container { background: #f5f5f5; padding: 20px; border-radius: 10px; margin: 10px 0; }
                button { background: #007bff; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer; margin: 5px; }
                button:hover { background: #0056b3; }
                input, textarea { width: 100%; padding: 10px; margin: 5px 0; border: 1px solid #ddd; border-radius: 5px; }
                #status { padding: 10px; margin: 10px 0; border-radius: 5px; }
                .success { background: #d4edda; color: #155724; }
                .error { background: #f8d7da; color: #721c24; }
            </style>
        </head>
        <body>
            <h1>🤖 Dia Real-Time Voice Agent</h1>
            
            <div class="container">
                <h3>Voice Reference Upload</h3>
                <input type="file" id="voiceFile" accept="audio/*">
                <button onclick="uploadVoiceReference()">Upload Voice Reference</button>
            </div>
            
            <div class="container">
                <h3>Text to Speech</h3>
                <textarea id="textInput" placeholder="Enter text to convert to speech..." rows="3"></textarea>
                <button onclick="generateSpeech()">Generate Speech</button>
                <audio id="audioPlayer" controls style="width: 100%; margin-top: 10px;"></audio>
            </div>
            
            <div class="container">
                <h3>Real-Time Conversation</h3>
                <input type="text" id="conversationInput" placeholder="Type your message...">
                <button onclick="sendMessage()">Send Message</button>
                <button onclick="connectWebSocket()">Connect WebSocket</button>
                <div id="status"></div>
            </div>
            
            <script>
                let ws = null;
                
                function showStatus(message, isError = false) {
                    const status = document.getElementById('status');
                    status.textContent = message;
                    status.className = isError ? 'error' : 'success';
                }
                
                async function uploadVoiceReference() {
                    const fileInput = document.getElementById('voiceFile');
                    const file = fileInput.files[0];
                    
                    if (!file) {
                        showStatus('Please select a file', true);
                        return;
                    }
                    
                    const formData = new FormData();
                    formData.append('file', file);
                    
                    try {
                        const response = await fetch('/upload-voice-reference', {
                            method: 'POST',
                            body: formData
                        });
                        
                        if (response.ok) {
                            showStatus('Voice reference uploaded successfully!');
                        } else {
                            showStatus('Upload failed', true);
                        }
                    } catch (error) {
                        showStatus('Upload error: ' + error.message, true);
                    }
                }
                
                async function generateSpeech() {
                    const text = document.getElementById('textInput').value;
                    
                    if (!text.trim()) {
                        showStatus('Please enter some text', true);
                        return;
                    }
                    
                    try {
                        const response = await fetch('/generate-speech?text=' + encodeURIComponent(text), {
                            method: 'POST'
                        });
                        
                        if (response.ok) {
                            const audioBlob = await response.blob();
                            const audioUrl = URL.createObjectURL(audioBlob);
                            document.getElementById('audioPlayer').src = audioUrl;
                            showStatus('Speech generated successfully!');
                        } else {
                            showStatus('Speech generation failed', true);
                        }
                    } catch (error) {
                        showStatus('Error: ' + error.message, true);
                    }
                }
                
                function connectWebSocket() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    ws = new WebSocket(protocol + '//' + window.location.host + '/ws');
                    
                    ws.onopen = function() {
                        showStatus('WebSocket connected!');
                    };
                    
                    ws.onmessage = function(event) {
                        const data = JSON.parse(event.data);
                        if (data.type === 'audio_response') {
                            const audioBlob = new Blob([Uint8Array.from(atob(data.audio), c => c.charCodeAt(0))], {type: 'audio/wav'});
                            const audioUrl = URL.createObjectURL(audioBlob);
                            document.getElementById('audioPlayer').src = audioUrl;
                            document.getElementById('audioPlayer').play();
                        }
                    };
                    
                    ws.onerror = function() {
                        showStatus('WebSocket error', true);
                    };
                    
                    ws.onclose = function() {
                        showStatus('WebSocket disconnected', true);
                    };
                }
                
                function sendMessage() {
                    const input = document.getElementById('conversationInput');
                    const message = input.value.trim();
                    
                    if (!message || !ws) {
                        showStatus('Please connect WebSocket and enter a message', true);
                        return;
                    }
                    
                    ws.send(JSON.stringify({
                        type: 'text_input',
                        text: message
                    }));
                    
                    input.value = '';
                    showStatus('Message sent, generating response...');
                }
                
                // Enter key support
                document.getElementById('conversationInput').addEventListener('keypress', function(e) {
                    if (e.key === 'Enter') {
                        sendMessage();
                    }
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
        
        uvicorn.run(self.app, host="0.0.0.0", port=self.port)

def main():
    """Main execution function"""
    print("🎯 Dia TTS Real-Time Conversational AI Agent")
    print("=" * 50)
    
    # Setup
    setup = DiaSetup()
    setup.complete_setup()
    
    # Initialize voice agent
    print("🤖 Initializing voice agent...")
    voice_agent = DiaVoiceAgent(
        voice_seed=42,  # Fixed seed for consistency
        reference_audio_path=None  # Set this to your voice reference file
    )
    
    # Load model
    voice_agent.load_model()
    
    # Create and run server
    server = RealTimeVoiceServer(voice_agent, port=8000)
    
    print("\n🎉 Setup complete! Features available:")
    print("✅ Single speaker consistency (fixed seed)")
    print("✅ Voice cloning support (10-second audio)")
    print("✅ Real-time conversation")
    print("✅ WebSocket support")
    print("✅ Web interface")
    print("✅ REST API endpoints")
    
    # Run server
    server.run()

if __name__ == "__main__":
    main()
