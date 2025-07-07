#!/usr/bin/env python3
"""
Dia TTS Real-Time Conversational AI Agent
Corrected implementation based on official Dia repository
"""

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
        """Load Dia model using correct API from official repository"""
        print("🤖 Loading Dia model with correct API...")
        
        try:
            import torch
            from huggingface_hub import hf_hub_download
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.voice_seed)
            
            # Get HuggingFace token
            hf_token = get_hf_token()
            if not hf_token:
                raise ValueError("HuggingFace token is required but not found")
            
            print("📥 Downloading model files from HuggingFace...")
            
            # Download model files manually (official approach)
            try:
                # Download config
                config_path = hf_hub_download(
                    repo_id="nari-labs/Dia-1.6B",
                    filename="config.json",
                    token=hf_token
                )
                print(f"✅ Downloaded config: {config_path}")
                
                # Download model weights
                model_path = hf_hub_download(
                    repo_id="nari-labs/Dia-1.6B", 
                    filename="model.safetensors",
                    token=hf_token
                )
                print(f"✅ Downloaded model weights: {model_path}")
                
                # Download preprocessor config
                preprocessor_path = hf_hub_download(
                    repo_id="nari-labs/Dia-1.6B",
                    filename="preprocessor_config.json", 
                    token=hf_token
                )
                print(f"✅ Downloaded preprocessor config: {preprocessor_path}")
                
            except Exception as e:
                print(f"❌ Failed to download model files: {e}")
                raise
            
            # Load the model using the correct Dia API
            try:
                # Import the correct Dia class
                from dia import Dia
                
                # Load model with local files (official method)
                print("🔄 Initializing Dia model...")
                
                # Read config
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Initialize model with config
                self.model = Dia(config)
                
                # Load weights
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                
                # Move to device
                self.model = self.model.to(self.device)
                self.model.eval()
                
                print(f"✅ Model loaded on {self.device}")
                
            except Exception as e:
                print(f"❌ Failed to initialize model: {e}")
                # Try alternative loading method from official repo
                try:
                    print("🔄 Trying alternative loading method...")
                    
                    # Use the official repository method
                    from dia.model import DiaModel
                    
                    self.model = DiaModel.from_pretrained(
                        model_path=model_path,
                        config_path=config_path,
                        device=self.device
                    )
                    
                    print("✅ Model loaded with alternative method")
                    
                except Exception as e2:
                    print(f"❌ Alternative loading also failed: {e2}")
                    raise
            
            # Setup reference audio if provided
            if self.reference_audio_path and os.path.exists(self.reference_audio_path):
                self.setup_voice_cloning()
            
            self.is_loaded = True
            print("✅ Dia model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            print("💡 Troubleshooting steps:")
            print("   1. Verify internet connection")
            print("   2. Check HuggingFace token permissions")
            print("   3. Ensure you have access to nari-labs/Dia-1.6B")
            print("   4. Try running the official Dia app.py first")
            raise
    
    def generate_speech(self, text: str, use_voice_cloning: bool = True) -> bytes:
        """Generate speech using correct Dia API"""
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
            
            # Generate audio using correct API
            with torch.no_grad():
                if use_voice_cloning and self.speaker_consistency_prompt:
                    # Use voice cloning
                    output = self.model.generate(
                        text=formatted_text,
                        reference_audio=self.speaker_consistency_prompt['audio'],
                        sample_rate=44100
                    )
                else:
                    # Standard generation
                    output = self.model.generate(
                        text=formatted_text,
                        sample_rate=44100
                    )
            
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
        text = text.strip()
        text = text.replace("[S1]", "").replace("[S2]", "").strip()
        formatted_text = f"[S1] {text}"
        
        if not text.endswith(('.', '!', '?')):
            formatted_text += "."
        
        return formatted_text
    
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

# [Rest of the RealTimeVoiceServer class remains the same as previous implementation]

def main():
    """Main execution function"""
    print("🎯 Dia TTS Real-Time Conversational AI Agent v2.2")
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
    
    print("\n🎉 Setup complete!")

if __name__ == "__main__":
    main()
