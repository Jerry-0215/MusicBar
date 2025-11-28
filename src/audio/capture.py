"""
Audio capture module for real-time system audio monitoring.
"""
import sounddevice as sd
import numpy as np
import threading
import time
from typing import Callable, List, Dict, Optional, Any
from dataclasses import dataclass
from ..utils.config import config

class AudioCaptureError(Exception):
    """Custom exception for audio capture errors."""
    pass

class AudioCapture:
    """Real-time audio capture and processing."""
    
    def __init__(self, callback: Optional[Callable[[np.ndarray, int], None]] = None):
        """
        Initialize audio capture.
        
        Args:
            callback: Function to call with audio data (audio_data, frames)
        """
        self.callback = callback
        self.is_running = False
        self.stream = None
        self._lock = threading.Lock()
        
        # Audio configuration
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.chunk_size = config.audio.chunk_size
        self.device = config.audio.device
        self.dtype = config.audio.dtype
        
        if config.debug:
            print(f"Audio config: {self.sample_rate}Hz, {self.channels}ch, chunk={self.chunk_size}")
    
    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags):
        """Internal callback for sounddevice stream."""
        if status:
            if config.debug:
                print(f"Audio callback status: {status}")
        
        if self.callback and self.is_running:
            try:
                # Convert to the format we want
                audio_data = indata.copy().astype(np.float32)
                self.callback(audio_data, frames)
            except Exception as e:
                if config.debug:
                    print(f"Error in audio callback: {e}")
    
    def start(self) -> bool:
        """
        Start audio capture.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        with self._lock:
            if self.is_running:
                return True
            
            try:
                # List available devices for debugging
                if config.debug:
                    print("Available audio devices:")
                    print(sd.query_devices())
                
                # Create input stream
                self.stream = sd.InputStream(
                    device=self.device,
                    channels=self.channels,
                    samplerate=self.sample_rate,
                    blocksize=self.chunk_size,
                    dtype=self.dtype,
                    callback=self._audio_callback
                )
                
                self.stream.start()
                self.is_running = True
                
                if config.debug:
                    print(f"Audio capture started successfully")
                    print(f"Using device: {self.stream.device}")
                
                return True
                
            except Exception as e:
                if config.debug:
                    print(f"Failed to start audio capture: {e}")
                raise AudioCaptureError(f"Failed to start audio capture: {e}")
    
    def stop(self):
        """Stop audio capture."""
        with self._lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            if self.stream:
                try:
                    self.stream.stop()
                    self.stream.close()
                except Exception as e:
                    if config.debug:
                        print(f"Error stopping audio stream: {e}")
                finally:
                    self.stream = None
            
            if config.debug:
                print("Audio capture stopped")
    
    def get_device_info(self) -> dict:
        """Get information about the current audio device."""
        try:
            if self.device is not None:
                return sd.query_devices(self.device)
            else:
                return sd.query_devices(kind='input')
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def list_devices() -> list:
        """List all available audio devices."""
        try:
            devices = sd.query_devices()
            device_list = []
            
            for i, device in enumerate(devices):
                # Include devices with input channels > 0
                if device["max_input_channels"] > 0:
                    device_list.append({
                        "id": i,
                        "name": device["name"],
                        "channels": device["max_input_channels"],
                        "sample_rate": device["default_samplerate"],
                        "type": "input"
                    })
                # Also include aggregate devices that might be useful for monitoring
                # (devices with "aggregate" or "blackhole" in the name)
                elif ("aggregate" in device["name"].lower() or 
                      "blackhole" in device["name"].lower() or
                      "bh +" in device["name"].lower()):
                    device_list.append({
                        "id": i,
                        "name": device["name"] + " (output only)",
                        "channels": 0,
                        "sample_rate": device["default_samplerate"],
                        "type": "output"
                    })
            
            if config.debug:
                print(f"Found {len(device_list)} audio devices:")
                for device in device_list:
                    print(f"  {device['id']}: {device['name']} ({device['channels']}ch input)")
            
            return device_list
        except Exception as e:
            if config.debug:
                print(f"Error listing devices: {e}")
            return []
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 