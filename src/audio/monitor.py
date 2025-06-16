"""
Audio monitoring and analysis module.
"""
import numpy as np
import threading
from collections import deque
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from utils.config import config
import librosa
import copy

@dataclass
class AudioMetrics:
    """Container for real-time audio metrics."""
    # Volume metrics
    overall_volume: float = 0.0
    left_volume: float = 0.0
    right_volume: float = 0.0
    
    # Stereo metrics
    stereo_width: float = 0.0  # How "wide" the stereo field is
    stereo_balance: float = 0.0  # -1.0 (left) to +1.0 (right)
    
    # Basic frequency info
    peak_frequency: float = 0.0
    spectral_centroid: float = 0.0
    frequency_history: deque = field(default_factory=lambda: deque(maxlen=25))
    
    # Musical features
    bpm: float = 0.0
    chord: str = "--"
    
    # Timing
    timestamp: float = 0.0
    
    # Volume history for visualization
    volume_history: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def __post_init__(self):
        """Initialize with proper config values after creation."""
        # Update maxlen if config is available
        try:
            # Replace with properly configured deque
            self.volume_history = deque(maxlen=config.ui.volume_history_length)
        except:
            # Fallback to default if config not available
            pass

class AudioMonitor:
    """Real-time audio analysis and monitoring."""
    
    _chord_templates: Dict[str, np.ndarray] = {}

    def __init__(self, update_callback: Optional[Callable[[AudioMetrics], None]] = None):
        """
        Initialize audio monitor.
        
        Args:
            update_callback: Function to call when metrics are updated
        """
        self.update_callback = update_callback
        self.metrics = AudioMetrics()
        self._lock = threading.Lock()
        
        # Configuration
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        
        # Buffer for musical analysis
        self.analysis_buffer_size = self.sample_rate * 5  # 5 seconds
        self.analysis_buffer = np.array([], dtype=np.float32)

        if not AudioMonitor._chord_templates:
            AudioMonitor._chord_templates = self._create_chord_templates()

        if config.debug:
            print("AudioMonitor initialized")
    
    @staticmethod
    def _create_chord_templates() -> Dict[str, np.ndarray]:
        """Create a dictionary of ideal chroma vectors for major/minor chords."""
        templates = {}
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i in range(12):
            # Major chord: root, major third, perfect fifth
            major_template = np.zeros(12)
            major_template[i] = 1.0  # Root
            major_template[(i + 4) % 12] = 0.8  # Major Third
            major_template[(i + 7) % 12] = 0.9  # Fifth
            templates[f"{pitch_classes[i]} Major"] = major_template / (np.linalg.norm(major_template) + 1e-9)
            
            # Minor chord: root, minor third, perfect fifth
            minor_template = np.zeros(12)
            minor_template[i] = 1.0  # Root
            minor_template[(i + 3) % 12] = 0.8  # Minor Third
            minor_template[(i + 7) % 12] = 0.9  # Fifth
            templates[f"{pitch_classes[i]} Minor"] = minor_template / (np.linalg.norm(minor_template) + 1e-9)
        return templates

    def process_audio(self, audio_data: np.ndarray, frames: int):
        """
        Process incoming audio data and update metrics.
        
        Args:
            audio_data: Audio samples as numpy array (frames, channels)
            frames: Number of audio frames
        """
        with self._lock:
            try:
                # Ensure we have stereo data
                if audio_data.ndim == 1:
                    # Mono to stereo
                    audio_data = np.column_stack([audio_data, audio_data])
                elif audio_data.shape[1] == 1:
                    # Mono to stereo
                    audio_data = np.column_stack([audio_data[:, 0], audio_data[:, 0]])
                
                # Extract left and right channels
                left_channel = audio_data[:, 0]
                right_channel = audio_data[:, 1]
                
                # Calculate volume metrics (RMS)
                self.metrics.left_volume = float(np.sqrt(np.mean(left_channel**2)))
                self.metrics.right_volume = float(np.sqrt(np.mean(right_channel**2)))
                self.metrics.overall_volume = float(np.sqrt(np.mean(audio_data**2)))
                
                # Calculate stereo metrics
                self._calculate_stereo_metrics(left_channel, right_channel)
                
                # Calculate basic frequency metrics
                self._calculate_frequency_metrics(audio_data)

                # Update analysis buffer
                mono_signal = audio_data.mean(axis=1)
                self.analysis_buffer = np.concatenate((self.analysis_buffer, mono_signal))
                if len(self.analysis_buffer) > self.analysis_buffer_size:
                    self.analysis_buffer = self.analysis_buffer[-self.analysis_buffer_size:]

                # Calculate musical features if buffer is sufficiently full
                if len(self.analysis_buffer) >= self.sample_rate * 2:  # Need at least 2s of audio
                    self._calculate_musical_features(mono_signal=self.analysis_buffer)
                else:
                    self.metrics.bpm = 0.0
                    self.metrics.chord = "--"
                
                # Update timestamp
                import time
                self.metrics.timestamp = time.time()
                
                # Update volume history
                self.metrics.volume_history.append(self.metrics.overall_volume)
                
                # Call update callback
                if self.update_callback:
                    self.update_callback(self.metrics)
                
            except Exception as e:
                if config.debug:
                    print(f"Error processing audio: {e}")
    
    def _calculate_stereo_metrics(self, left: np.ndarray, right: np.ndarray):
        """Calculate stereo field metrics."""
        # Stereo balance: difference between left and right
        left_energy = np.mean(left**2)
        right_energy = np.mean(right**2)
        total_energy = left_energy + right_energy
        
        if total_energy > 0:
            self.metrics.stereo_balance = float((right_energy - left_energy) / total_energy)
        else:
            self.metrics.stereo_balance = 0.0
        
        # Stereo width: correlation between channels (inverted)
        # High correlation = narrow stereo field
        # Low correlation = wide stereo field
        try:
            correlation = np.corrcoef(left, right)[0, 1]
            if np.isnan(correlation):
                correlation = 1.0
            self.metrics.stereo_width = float(1.0 - abs(correlation))
        except:
            self.metrics.stereo_width = 0.0
    
    def _calculate_frequency_metrics(self, audio_data: np.ndarray):
        """Calculate basic frequency domain metrics."""
        try:
            # Take FFT of the audio data
            mono_signal = np.mean(audio_data, axis=1)  # Convert to mono
            fft = np.fft.rfft(mono_signal)
            magnitude = np.abs(fft)
            
            # Find peak frequency
            peak_idx = np.argmax(magnitude)
            freqs = np.fft.rfftfreq(len(mono_signal), 1/self.sample_rate)
            peak_freq = float(freqs[peak_idx])
            self.metrics.peak_frequency = peak_freq
            # Only add meaningful frequencies to the history
            if peak_freq > 1.0:
                self.metrics.frequency_history.append(peak_freq)
            
            # Calculate spectral centroid (brightness)
            if np.sum(magnitude) > 0:
                self.metrics.spectral_centroid = float(
                    np.sum(freqs * magnitude) / np.sum(magnitude)
                )
            else:
                self.metrics.spectral_centroid = 0.0
                
        except Exception as e:
            if config.debug:
                print(f"Error calculating frequency metrics: {e}")
            self.metrics.peak_frequency = 0.0
            self.metrics.spectral_centroid = 0.0
    
    def _calculate_musical_features(self, mono_signal: np.ndarray):
        """Calculate musical features like BPM and chord from a larger buffer."""
        try:
            # --- BPM Detection ---
            tempo, _ = librosa.beat.beat_track(y=mono_signal, sr=self.sample_rate)
            if tempo > 0:
                self.metrics.bpm = float(tempo)
            else:
                self.metrics.bpm = 0.0

            # --- Chord Detection ---
            chroma = librosa.feature.chroma_stft(y=mono_signal, sr=self.sample_rate)
            chroma_mean = np.mean(chroma, axis=1)

            # Avoid division by zero if audio is silent
            if np.sum(chroma_mean) > 0:
                chroma_normalized = chroma_mean / np.linalg.norm(chroma_mean)
            else:
                self.metrics.chord = "--"
                return

            best_match_score = -1
            best_match_chord = "--"
            for chord_name, template_normalized in self._chord_templates.items():
                score = np.dot(chroma_normalized, template_normalized) # Cosine similarity
                if score > best_match_score:
                    best_match_score = score
                    best_match_chord = chord_name
            
            if config.debug:
                print(f"Chord detection: Best match '{best_match_chord}' with score {best_match_score:.2f}")
            
            self.metrics.chord = best_match_chord

        except Exception as e:
            if config.debug:
                print(f"Error calculating musical features: {e}")
            self.metrics.bpm = 0.0
            self.metrics.chord = "--"
    
    def get_metrics(self) -> AudioMetrics:
        """Get current audio metrics (thread-safe)."""
        with self._lock:
            # Return a copy to ensure thread safety for the caller
            return copy.copy(self.metrics)
    
    def get_volume_db(self) -> Dict[str, float]:
        """Get volume levels in dB."""
        with self._lock:
            def linear_to_db(linear_vol: float) -> float:
                if linear_vol > 0:
                    return 20 * np.log10(linear_vol)
                else:
                    return -60.0  # Silence threshold
            
            return {
                "overall_db": linear_to_db(self.metrics.overall_volume),
                "left_db": linear_to_db(self.metrics.left_volume),
                "right_db": linear_to_db(self.metrics.right_volume)
            }
    
    def get_stereo_info(self) -> Dict[str, Any]:
        """Get detailed stereo field information."""
        with self._lock:
            balance_description = "Center"
            if self.metrics.stereo_balance < -0.1:
                balance_description = "Left"
            elif self.metrics.stereo_balance > 0.1:
                balance_description = "Right"
            
            width_description = "Narrow"
            if self.metrics.stereo_width > 0.5:
                width_description = "Wide"
            elif self.metrics.stereo_width > 0.3:
                width_description = "Medium"
            
            return {
                "balance": self.metrics.stereo_balance,
                "balance_description": balance_description,
                "width": self.metrics.stereo_width,
                "width_description": width_description,
                "left_volume": self.metrics.left_volume,
                "right_volume": self.metrics.right_volume
            }
    
    def reset_metrics(self):
        """Reset all metrics to default values."""
        with self._lock:
            self.metrics = AudioMetrics()
            if config.debug:
                print("Audio metrics reset") 