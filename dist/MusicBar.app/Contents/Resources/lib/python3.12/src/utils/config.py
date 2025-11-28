"""
Configuration management for MusicBar app.
"""
import os
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AudioConfig:
    """Audio capture and processing configuration."""
    sample_rate: int = 44100  # CD quality
    channels: int = 2  # Stereo
    chunk_size: int = 2048  # Buffer size for real-time processing
    device: Optional[int] = None  # None = default device
    dtype: str = 'float32'  # Audio data type
    
    # Analysis buffer size in seconds
    analysis_buffer_size: int = 5
    
    # How often to run heavy analysis
    bpm_analysis_interval: float = 2.0
    chord_analysis_interval: float = 1.0
    
    # Confidence threshold for chord detection
    chord_confidence_threshold: float = 0.4
    
@dataclass
class UIConfig:
    """User interface configuration."""
    update_interval: float = 0.1  # Update frequency in seconds
    show_notifications: bool = True
    volume_history_size: int = 20
    
@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = "MusicBar"
    version: str = "0.1.0"
    debug: bool = True  # Enable debug for troubleshooting
    audio: AudioConfig = field(default_factory=AudioConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    @classmethod
    def load_from_env(cls) -> 'AppConfig':
        """Load configuration from environment variables."""
        config = cls()
        
        # Audio settings
        if os.getenv('MUSICBAR_SAMPLE_RATE'):
            config.audio.sample_rate = int(os.getenv('MUSICBAR_SAMPLE_RATE'))
        if os.getenv('MUSICBAR_CHANNELS'):
            config.audio.channels = int(os.getenv('MUSICBAR_CHANNELS'))
        if os.getenv('MUSICBAR_CHUNK_SIZE'):
            config.audio.chunk_size = int(os.getenv('MUSICBAR_CHUNK_SIZE'))
        if os.getenv('MUSICBAR_DEVICE'):
            config.audio.device = int(os.getenv('MUSICBAR_DEVICE'))
            
        # UI settings
        if os.getenv('MUSICBAR_DEBUG'):
            config.debug = os.getenv('MUSICBAR_DEBUG').lower() == 'true'
            
        return config

# Global configuration instance
config = AppConfig.load_from_env() 