"""
System audio capture using AVAudioEngine and Screen Recording permission.
This allows capturing system audio directly without BlackHole.
"""
import numpy as np
import threading
import time
from typing import Callable, Optional, Any
import objc
from Foundation import NSObject, NSRunLoop, NSDefaultRunLoopMode
from AVFoundation import (
    AVAudioEngine, AVAudioInputNode, AVAudioFormat, AVAudioPCMBuffer,
    AVAudioSession, AVAudioSessionCategoryPlayAndRecord, 
    AVAudioSessionCategoryOptionDefaultToSpeaker,
    AVAudioSessionCategoryOptionAllowBluetooth
)
from ..utils.config import config

class SystemAudioCaptureError(Exception):
    """Custom exception for system audio capture errors."""
    pass

class SystemAudioCapture:
    """System audio capture using AVAudioEngine."""
    
    def __init__(self, callback: Optional[Callable[[np.ndarray, int], None]] = None):
        """
        Initialize system audio capture.
        
        Args:
            callback: Function to call with audio data (audio_data, frames)
        """
        self.callback = callback
        self.is_running = False
        self.audio_engine = None
        self.input_node = None
        self._lock = threading.Lock()
        
        # Audio configuration
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.chunk_size = config.audio.chunk_size
        
        if config.debug:
            print(f"SystemAudioCapture initialized: {self.sample_rate}Hz, {self.channels}ch")
    
    def request_screen_recording_permission(self) -> bool:
        """
        Request Screen Recording permission which allows system audio access.
        """
        try:
            # Import required frameworks
            from Quartz import CGRequestScreenCaptureAccess
            from AppKit import NSAlert, NSAlertFirstButtonReturn
            
            # Check if we already have permission
            if self._has_screen_recording_permission():
                if config.debug:
                    print("✅ Screen Recording permission already granted")
                return True
            
            # Request permission
            if config.debug:
                print("🔐 Requesting Screen Recording permission...")
            
            # Show alert to user
            alert = NSAlert.alloc().init()
            alert.setMessageText_("Screen Recording Permission Required")
            alert.setInformativeText_(
                "MusicBar needs Screen Recording permission to capture system audio. "
                "Please grant permission in System Preferences when prompted."
            )
            alert.addButtonWithTitle_("OK")
            alert.runModal()
            
            # Request the permission
            permission_granted = CGRequestScreenCaptureAccess()
            
            if permission_granted:
                if config.debug:
                    print("✅ Screen Recording permission granted")
                return True
            else:
                if config.debug:
                    print("❌ Screen Recording permission denied")
                return False
                
        except Exception as e:
            if config.debug:
                print(f"Error requesting Screen Recording permission: {e}")
            return False
    
    def _has_screen_recording_permission(self) -> bool:
        """Check if we already have Screen Recording permission."""
        try:
            from Quartz import CGPreflightScreenCaptureAccess
            return CGPreflightScreenCaptureAccess()
        except:
            return False
    
    def _setup_audio_session(self):
        """Set up the audio session for system audio capture."""
        try:
            session = AVAudioSession.sharedInstance()
            
            # Set category for recording
            error = None
            success = session.setCategoryWithOptions_error_(
                AVAudioSessionCategoryPlayAndRecord,
                AVAudioSessionCategoryOptionDefaultToSpeaker | 
                AVAudioSessionCategoryOptionAllowBluetooth,
                error
            )
            
            if not success:
                raise SystemAudioCaptureError(f"Failed to set audio session category: {error}")
            
            # Activate the session
            success = session.setActive_error_(True, error)
            if not success:
                raise SystemAudioCaptureError(f"Failed to activate audio session: {error}")
                
            if config.debug:
                print("✅ Audio session configured")
                
        except Exception as e:
            raise SystemAudioCaptureError(f"Failed to setup audio session: {e}")
    
    def _audio_callback(self, buffer, when):
        """Callback for processing audio data from AVAudioEngine."""
        if not self.is_running or not self.callback:
            return
        
        try:
            # Extract audio data from AVAudioPCMBuffer
            frame_count = int(buffer.frameLength())
            channel_count = int(buffer.format().channelCount())
            
            if frame_count == 0:
                return
            
            # Get the audio data as numpy array
            # AVAudioPCMBuffer provides access to raw audio data
            audio_data = self._buffer_to_numpy(buffer, frame_count, channel_count)
            
            if audio_data is not None:
                self.callback(audio_data, frame_count)
                
        except Exception as e:
            if config.debug:
                print(f"Error in audio callback: {e}")
    
    def _buffer_to_numpy(self, buffer, frame_count: int, channel_count: int) -> Optional[np.ndarray]:
        """Convert AVAudioPCMBuffer to numpy array."""
        try:
            # Get pointer to the audio data
            float_channel_data = buffer.floatChannelData()
            if not float_channel_data:
                return None
            
            # Convert to numpy array
            audio_data = np.zeros((frame_count, channel_count), dtype=np.float32)
            
            for channel in range(channel_count):
                if channel < buffer.format().channelCount():
                    # Get pointer to channel data
                    channel_ptr = float_channel_data[channel]
                    # Convert to numpy array
                    channel_data = np.frombuffer(
                        (objc.PyObjC_PythonToId(channel_ptr) * frame_count).contents,
                        dtype=np.float32,
                        count=frame_count
                    )
                    audio_data[:, channel] = channel_data
            
            return audio_data
            
        except Exception as e:
            if config.debug:
                print(f"Error converting buffer to numpy: {e}")
            return None
    
    def start(self) -> bool:
        """
        Start system audio capture.
        
        Returns:
            bool: True if started successfully, False otherwise
        """
        with self._lock:
            if self.is_running:
                return True
            
            try:
                # Check for Screen Recording permission
                if not self._has_screen_recording_permission():
                    if not self.request_screen_recording_permission():
                        raise SystemAudioCaptureError(
                            "Screen Recording permission required for system audio capture"
                        )
                
                # Set up audio session
                self._setup_audio_session()
                
                # Create and configure AVAudioEngine
                self.audio_engine = AVAudioEngine.alloc().init()
                self.input_node = self.audio_engine.inputNode()
                
                # Set up the audio format
                input_format = self.input_node.inputFormatForBus_(0)
                
                if config.debug:
                    print(f"Input format: {input_format.sampleRate()}Hz, {input_format.channelCount()}ch")
                
                # Install tap on input node to capture audio
                self.input_node.installTapOnBus_bufferSize_format_block_(
                    0,  # bus
                    self.chunk_size,  # buffer size
                    input_format,  # format
                    self._audio_callback  # callback block
                )
                
                # Prepare and start the engine
                self.audio_engine.prepare()
                error = None
                success = self.audio_engine.startAndReturnError_(error)
                
                if not success:
                    raise SystemAudioCaptureError(f"Failed to start audio engine: {error}")
                
                self.is_running = True
                
                if config.debug:
                    print("✅ System audio capture started successfully")
                
                return True
                
            except Exception as e:
                if config.debug:
                    print(f"Failed to start system audio capture: {e}")
                raise SystemAudioCaptureError(f"Failed to start system audio capture: {e}")
    
    def stop(self):
        """Stop system audio capture."""
        with self._lock:
            if not self.is_running:
                return
            
            self.is_running = False
            
            try:
                if self.input_node:
                    self.input_node.removeTapOnBus_(0)
                    self.input_node = None
                
                if self.audio_engine:
                    self.audio_engine.stop()
                    self.audio_engine = None
                
                if config.debug:
                    print("✅ System audio capture stopped")
                    
            except Exception as e:
                if config.debug:
                    print(f"Error stopping system audio capture: {e}")
    
    def get_device_info(self) -> dict:
        """Get information about the system audio capture."""
        return {
            "name": "System Audio (Screen Recording)",
            "type": "system",
            "channels": self.channels,
            "sample_rate": self.sample_rate,
            "permission_required": "Screen Recording"
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 