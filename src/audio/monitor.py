"""
Audio monitoring and analysis module using Essentia.
"""
import numpy as np
import threading
from collections import deque
from typing import Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from utils.config import config
import time

# Essentia imports
import essentia
import essentia.standard as es

@dataclass
class AudioMetrics:
    """Dataclass to hold all calculated audio metrics."""
    volume_history: deque = field(default_factory=lambda: deque(maxlen=config.ui.volume_history_size))
    stereo_balance: float = 0.0
    stereo_width: float = 0.0
    bpm: float = 0.0
    chord: str = "N/A"
    confidence: float = 0.0
    
    def copy(self):
        return AudioMetrics(
            volume_history=self.volume_history.copy(),
            stereo_balance=self.stereo_balance,
            stereo_width=self.stereo_width,
            bpm=self.bpm,
            chord=self.chord,
            confidence=self.confidence,
        )

class AudioMonitor:
    """
    Analyzes audio data in real-time using Essentia to extract musical features.
    """
    _chord_templates: Dict[str, np.ndarray] = {}
    
    def __init__(self, update_callback=None):
        self.sample_rate = config.audio.sample_rate
        self.channels = config.audio.channels
        self.update_callback = update_callback
        
        self._lock = threading.Lock()
        self._metrics = AudioMetrics()
        
        # Buffer for analysis that requires a longer context
        buffer_size_seconds = config.audio.analysis_buffer_size
        self.analysis_buffer_len = int(buffer_size_seconds * self.sample_rate)
        self._analysis_buffer = np.array([], dtype=np.float32)
        
        # Essentia Algorithms
        self.frame_size = 4096 # Larger frame for better freq resolution
        self.hop_size = 2048
        
        # More robust BPM estimator
        self.rhythm_extractor = es.RhythmExtractor2013(method="multifeature")
        
        # Algorithms for chromagram-based chord detection
        self.window = es.Windowing(type='hann')
        self.spectrum = es.Spectrum()
        self.spectral_peaks = es.SpectralPeaks(
            magnitudeThreshold=0.00001,
            minFrequency=20,
            maxFrequency=3500,
            maxPeaks=60
        )
        self.hpcp = es.HPCP()

        if not AudioMonitor._chord_templates:
            AudioMonitor._chord_templates = self._create_chord_templates()

        self.last_analysis_time = 0
        self.analysis_interval = config.audio.analysis_interval

        if config.debug:
            print("AudioMonitor (Essentia) initialized with new algorithms")

    @staticmethod
    def _create_chord_templates() -> Dict[str, np.ndarray]:
        """Create a dictionary of ideal chroma vectors for major/minor chords."""
        templates = {}
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        for i in range(12):
            major_template = np.zeros(12)
            major_template[i] = 1.0
            major_template[(i + 4) % 12] = 0.8
            major_template[(i + 7) % 12] = 0.9
            templates[f"{pitch_classes[i]} Major"] = major_template / (np.linalg.norm(major_template) or 1e-9)
            
            minor_template = np.zeros(12)
            minor_template[i] = 1.0
            minor_template[(i + 3) % 12] = 0.8
            minor_template[(i + 7) % 12] = 0.9
            templates[f"{pitch_classes[i]} Minor"] = minor_template / (np.linalg.norm(minor_template) or 1e-9)
        return templates

    def process_audio(self, audio_data: np.ndarray, frames: int):
        """
        Callback to receive audio data from the capture source.
        """
        if self.channels == 2:
            # Essentia algorithms work on mono. Average the channels.
            mono_data = np.mean(audio_data, axis=1)
        else:
            mono_data = audio_data

        # Essentia expects float32
        mono_data_float32 = mono_data.astype(np.float32)

        # Append to analysis buffer
        self._analysis_buffer = np.concatenate((self._analysis_buffer, mono_data_float32))
        if len(self._analysis_buffer) > self.analysis_buffer_len:
            self._analysis_buffer = self._analysis_buffer[-self.analysis_buffer_len:]

        # --- Perform immediate analysis (low-latency metrics) ---
        # Volume (RMS)
        rms = np.sqrt(np.mean(mono_data_float32**2))
        
        # Stereo info (from original stereo data)
        balance, width = 0.0, 0.0
        if self.channels == 2:
            left, right = audio_data[:, 0], audio_data[:, 1]
            balance = np.mean(left) - np.mean(right)
            width = np.mean(np.abs(left - right))

        # --- Perform buffered analysis (high-latency metrics) ---
        current_time = time.time()
        if current_time - self.last_analysis_time > self.analysis_interval:
            self.last_analysis_time = current_time
            # Run analysis in a separate thread to not block the audio callback
            threading.Thread(target=self._analyze_buffer).start()

        # Update metrics atomically
        with self._lock:
            self._metrics.volume_history.append(rms)
            self._metrics.stereo_balance = balance
            self._metrics.stereo_width = width
            if self.update_callback:
                self.update_callback(self._metrics.copy())

    def _analyze_buffer(self):
        """
        Analyze the content of the buffer for features like BPM and chords.
        This is called periodically, not on every audio chunk.
        """
        # Create a copy for analysis to avoid race conditions
        with self._lock:
            analysis_data = self._analysis_buffer.copy()

        if len(analysis_data) < self.frame_size:
            return # Not enough data

        # --- BPM Detection ---
        try:
            bpm_info = self.rhythm_extractor(analysis_data)
            bpm = bpm_info[0]
        except Exception as e:
            if config.debug: print(f"[Essentia] BPM detection failed: {e}")
            bpm = 0.0
            
        # --- Chord Detection (Chromagram-based) ---
        try:
            chroma_frames = []
            for frame in es.FrameGenerator(analysis_data, frameSize=self.frame_size, hopSize=self.hop_size, startFromZero=True):
                freqs, magnitudes = self.spectral_peaks(self.spectrum(self.window(frame)))
                if not len(freqs): continue
                chroma_frames.append(self.hpcp(freqs, magnitudes))
            
            if not chroma_frames: raise Exception("No valid chroma frames")

            chroma_mean = np.mean(chroma_frames, axis=0)
            chroma_normalized = chroma_mean / (np.linalg.norm(chroma_mean) or 1e-9)

            best_match_score = -1
            best_chord = "N/A"
            for chord_name, template in self._chord_templates.items():
                score = np.dot(chroma_normalized, template)
                if score > best_match_score:
                    best_match_score = score
                    best_chord = chord_name
            
            confidence = best_match_score

        except Exception as e:
            if config.debug: print(f"[Essentia] Chord detection failed: {e}")
            best_chord, confidence = "N/A", 0.0

        # Update metrics atomically
        with self._lock:
            self._metrics.bpm = bpm
            if confidence > config.audio.chord_confidence_threshold:
                 self._metrics.chord = best_chord
                 self._metrics.confidence = confidence
            else:
                 self._metrics.chord = "N/A"
                 self._metrics.confidence = 0.0


    def get_metrics(self) -> AudioMetrics:
        """Get a thread-safe copy of the latest audio metrics."""
        with self._lock:
            return self._metrics.copy()
            
    def get_stereo_info(self) -> dict:
        """
        Provides descriptive information about the stereo image.
        """
        with self._lock:
            balance = self._metrics.stereo_balance
            width = self._metrics.stereo_width
        
        # Balance description
        if abs(balance) < 0.01: balance_desc = "Center"
        elif balance > 0: balance_desc = f"{(balance*100):.0f}% Right"
        else: balance_desc = f"{(-balance*100):.0f}% Left"
        
        # Width description
        if width < 0.1: width_desc = "Mono"
        elif width < 0.5: width_desc = "Narrow"
        elif width < 1.0: width_desc = "Stereo"
        else: width_desc = "Wide"
        
        return {
            "balance": balance,
            "width": width,
            "balance_description": balance_desc,
            "width_description": width_desc
        }

    def get_volume_db(self) -> Dict[str, float]:
        """Get volume levels in dB."""
        with self._lock:
            def linear_to_db(linear_vol: float) -> float:
                if linear_vol > 0:
                    return 20 * np.log10(linear_vol)
                else:
                    return -60.0  # Silence threshold
            
            return {
                "overall_db": linear_to_db(self._metrics.volume_history[-1]),
                "left_db": linear_to_db(self._metrics.volume_history[-1] / 2),
                "right_db": linear_to_db(self._metrics.volume_history[-1] / 2)
            }
    
    def reset_metrics(self):
        """Reset all metrics to default values."""
        with self._lock:
            self._metrics = AudioMetrics()
            if config.debug:
                print("Audio metrics reset") 