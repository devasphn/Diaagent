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
        self.device = "cuda" if self._check_cuda() else "cpu"
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        
    def load_model(self):
        """Load Dia model with correct API"""
        print("🤖 Loading Dia model...")
        
        try:
            # Try different import methods
            try:
                from dia import Dia
                print("✅ Imported from 'dia' module")
            except ImportError:
                try:
                    from nari_tts import Dia
                    print("✅ Imported from 'nari_tts' module")
                except ImportError:
                    # Try direct import from the cloned repo
                    sys.path.append('./dia')
                    from dia import Dia
                    print("✅ Imported from local 'dia' directory")
            
            import torch
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.voice_seed)
            
            # Load model with correct API
            print("📥 Downloading model from HuggingFace...")
            
            # Check if token is available
            token = os.getenv("HF_TOKEN")
            if token:
                print("✅ HF_TOKEN found")
                self.model = Dia.from_pretrained("nari-labs/Dia-1.6B", token=token)
            else:
                print("⚠️  No HF_TOKEN found, trying without token...")
                self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
            
            # Move to GPU if available
            if self.device == "cuda":
                self.model = self.model.to("cuda")
                print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                print("⚠️  Model loaded on CPU (GPU not available)")
            
            # Setup reference audio if provided
            if self.reference_audio_path and os.path.exists(self.reference_audio_path):
                self.setup_voice_cloning()
            
            self.is_loaded = True
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("💡 Troubleshooting steps:")
            print("   1. Make sure you ran: pip install -e . in the dia directory")
            print("   2. Set HF_TOKEN: export HF_TOKEN=your_token_here")
            print("   3. Check internet connection")
            raise
    
    def setup_voice_cloning(self):
        """Setup voice cloning with reference audio"""
        print("🎤 Setting up voice cloning...")
        
        try:
            import soundfile as sf
            
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
                "transcript": "This is my voice reference for cloning."
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
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.voice_seed)
            
            # Format text for single speaker
            formatted_text = self.format_text_for_single_speaker(text)
            
            # Generate audio with correct API
            if use_voice_cloning and self.speaker_consistency_prompt:
                # Use voice cloning with reference audio
                output = self.model.generate(
                    formatted_text,
                    reference_audio=self.speaker_consistency_prompt['audio']
                )
            else:
                # Standard generation with fixed seed
                output = self.model.generate(formatted_text)
            
            # Convert to bytes
            buffer = io.BytesIO()
            
            # Ensure output is numpy array
            if hasattr(output, 'cpu'):
                output = output.cpu().numpy()
            
            # Write as WAV
            sf.write(buffer, output, 44100, format='WAV')
            return buffer.getvalue()
            
        except Exception as e:
            print(f"❌ Speech generation failed: {e}")
            # Return empty audio on failure
            import numpy as np
            import soundfile as sf
            silence = np.zeros(44100)  # 1 second of silence
            buffer = io.BytesIO()
            sf.write(buffer, silence, 44100, format='WAV')
            return buffer.getvalue()
    
    def format_text_for_single_speaker(self, text: str) -> str:
        """Format text for consistent single speaker output"""
        # Clean and format text
        text = text.strip()
        
        # Remove any existing speaker tags
        text = text.replace("[S1]", "").replace("[S2]", "").strip()
        
        # Add single speaker tag
        formatted_text = f"[S1] {text}"
        
        # Ensure proper ending
        if not text.endswith(('.', '!', '?')):
            formatted_text += "."
        
        return formatted_text
    
    def process_conversation_turn(self, user_input: str) -> bytes:
        """Process a conversation turn and return audio response"""
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_input})
        
        # Generate response (integrate with your LLM here)
        response_text = self.generate_response(user_input)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Generate speech
        audio_bytes = self.generate_speech(response_text)
        
        return audio_bytes
    
    def generate_response(self, user_input: str) -> str:
        """Generate text response (replace with your LLM integration)"""
        # Simple responses for demonstration
        responses = [
            f"I understand you said: {user_input}. That's interesting!",
            f"Thank you for sharing that. Could you tell me more about {user_input}?",
            f"That's a great point about {user_input}. What do you think about it?",
            f"I appreciate your input on {user_input}. How does that make you feel?",
            f"Regarding {user_input}, I find that fascinating. Can you elaborate?",
            f"You mentioned {user_input}. That reminds me of something similar.",
        ]
        
        # Use seed for consistent but varied responses
        random.seed(hash(user_input) + self.voice_seed)
        return random.choice(responses)

# ... (rest of the RealTimeVoiceServer class remains the same)

def main():
    """Main execution function"""
    print("🎯 Dia TTS Real-Time Conversational AI Agent v2.0")
    print("=" * 60)
    
    # Set environment variables
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Check HuggingFace token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("⚠️  HF_TOKEN not set. You may need to set it for model access.")
        print("💡 Run: export HF_TOKEN=your_token_here")
        print("💡 Get your token from: https://huggingface.co/settings/tokens")
    else:
        print("✅ HF_TOKEN is set")
    
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
        print("💡 Make sure you have internet connection and model access")
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
    print("✅ Health monitoring")
    
    # Run server
    try:
        server.run()
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()
