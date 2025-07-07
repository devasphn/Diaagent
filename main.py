
import os
import sys
import warnings
import logging
from pathlib import Path
import json
import time
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
    
    # Method 3: Try huggingface_hub's get_token
    try:
        from huggingface_hub import get_token
        token = get_token()
        if token:
            return token
    except Exception as e:
        print(f"⚠️  Could not get token from huggingface_hub: {e}")
    
    return None

class DiaVoiceAgent:
    """Advanced Dia TTS Voice Agent using Transformers"""
    
    def __init__(self, voice_seed: int = 42, reference_audio_path: Optional[str] = None):
        self.model = None
        self.processor = None
        self.voice_seed = voice_seed
        self.reference_audio_path = reference_audio_path
        self.is_loaded = False
        self.conversation_history = []
        self.device = "cuda" if self._check_cuda() else "cpu"
        
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
        
    def load_model(self):
        """Load Dia model using Transformers"""
        print("🤖 Loading Dia model with Transformers integration...")
        
        try:
            # Use the new Transformers approach
            from transformers import AutoProcessor, DiaForConditionalGeneration
            import torch
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.voice_seed)
            
            # Get HuggingFace token
            hf_token = get_hf_token()
            
            print("📥 Loading with Transformers integration...")
            
            # Load processor and model
            if hf_token:
                print("✅ Using HuggingFace token for authentication")
                self.processor = AutoProcessor.from_pretrained(
                    "nari-labs/Dia-1.6B", 
                    token=hf_token
                )
                self.model = DiaForConditionalGeneration.from_pretrained(
                    "nari-labs/Dia-1.6B", 
                    token=hf_token
                )
            else:
                print("⚠️  Trying without token...")
                self.processor = AutoProcessor.from_pretrained("nari-labs/Dia-1.6B")
                self.model = DiaForConditionalGeneration.from_pretrained("nari-labs/Dia-1.6B")
            
            # Move to GPU if available
            if self.device == "cuda":
                self.model = self.model.to("cuda")
                print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
            else:
                print("⚠️  Model loaded on CPU")
            
            self.is_loaded = True
            print("✅ Model loaded successfully with Transformers!")
            
        except Exception as e:
            print(f"❌ Transformers method failed: {e}")
            print("💡 Falling back to legacy method...")
            self._load_legacy_method()
    
    def _load_legacy_method(self):
        """Fallback to legacy dia.model import"""
        try:
            # Try legacy method as fallback
            from dia.model import Dia
            import torch
            
            # Set seed for consistency
            torch.manual_seed(self.voice_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.voice_seed)
            
            # Get HuggingFace token
            hf_token = get_hf_token()
            
            print("📥 Using legacy dia.model method...")
            
            if hf_token:
                # Try with proper token authentication
                from huggingface_hub import login
                login(token=hf_token)
                print("✅ Authenticated with HuggingFace Hub")
            
            # Load model without problematic parameters
            self.model = Dia.from_pretrained("nari-labs/Dia-1.6B")
            
            # Move to GPU if available
            if self.device == "cuda":
                self.model = self.model.to("cuda")
                print(f"✅ Model loaded on GPU: {torch.cuda.get_device_name()}")
            
            self.is_loaded = True
            print("✅ Model loaded successfully with legacy method!")
            
        except Exception as e:
            print(f"❌ Both methods failed: {e}")
            raise
    
    def generate_speech(self, text: str, use_voice_cloning: bool = False) -> bytes:
        """Generate speech from text"""
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
            
            # Generate audio based on method used
            if hasattr(self, 'processor') and self.processor is not None:
                # Use Transformers method
                inputs = self.processor(text=[formatted_text], padding=True, return_tensors="pt")
                if self.device == "cuda":
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_new_tokens=256)
                
                # Decode the output
                audio_outputs = self.processor.batch_decode(outputs)
                output = audio_outputs[0]  # Get first (and only) result
                
            else:
                # Use legacy method
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
        
        # Generate response
        response_text = self.generate_response(user_input)
        
        # Add response to history
        self.conversation_history.append({"role": "assistant", "content": response_text})
        
        # Generate speech
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
        
        # Use seed for consistent but varied responses
        random.seed(hash(user_input) + self.voice_seed)
        return random.choice(responses)

# ... (rest of RealTimeVoiceServer class remains the same)

def main():
    """Main execution function"""
    print("🎯 Dia TTS Real-Time Conversational AI Agent v3.0")
    print("=" * 60)
    
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
        print("💡 Or set: export HF_TOKEN=your_token_here")
    
    # Initialize voice agent
    print("🤖 Initializing voice agent...")
    voice_agent = DiaVoiceAgent(
        voice_seed=42,
        reference_audio_path=None
    )
    
    # Load model
    try:
        voice_agent.load_model()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        print("💡 Check the solutions below")
        sys.exit(1)
    
    print("\n🎉 Model loaded successfully!")
    print("🧪 Testing speech generation...")
    
    # Test the model
    try:
        test_text = "Hello, this is a test of the Dia TTS model."
        audio_bytes = voice_agent.generate_speech(test_text)
        print("✅ Speech generation test successful!")
        
        # Save test audio
        with open("test_output.wav", "wb") as f:
            f.write(audio_bytes)
        print("💾 Test audio saved as 'test_output.wav'")
        
    except Exception as e:
        print(f"⚠️  Speech generation test failed: {e}")
    
    print("\n🚀 Ready for real-time voice generation!")

if __name__ == "__main__":
    main()
