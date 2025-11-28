"""
Professional instrument classification using librosa.
This module provides much more accurate instrument detection than the basic spectral approach.
"""
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from ..utils.config import config

class InstrumentClassifier:
    """
    Professional instrument classification using librosa.
    Provides much better accuracy than basic spectral analysis.
    """
    
    def __init__(self):
        self.sample_rate = config.audio.sample_rate
        
        # Pre-defined instrument classes
        self.instrument_classes = [
            "Piano", "Guitar", "Violin", "Drums", "Bass", "Vocals",
            "Saxophone", "Trumpet", "Flute", "Cello", "Harmonica"
        ]
        
        if config.debug:
            print("InstrumentClassifier initialized with librosa")
    
    def classify_audio(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Classify instruments in audio data using librosa.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary of instrument names and confidence scores
        """
        try:
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ensure audio is float32
            audio_data = audio_data.astype(np.float32)
            
            # Extract features using librosa
            features = self._extract_librosa_features(audio_data)
            
            # Classify using the features
            instrument_scores = self._classify_from_librosa_features(features)
            
            return instrument_scores
            
        except Exception as e:
            if config.debug:
                print(f"Instrument classification failed: {e}")
            return {"Unknown": 0.0}
    
    def _extract_librosa_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract features using librosa."""
        features = {}
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data)[0]
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        
        # RMS energy
        rms = librosa.feature.rms(y=audio_data)[0]
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate)
        
        # Tonnetz (tonal centroid features)
        tonnetz = librosa.feature.tonnetz(y=audio_data, sr=self.sample_rate)
        
        # Average features across time
        features['spectral_centroid'] = np.mean(spectral_centroids)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        features['zero_crossing_rate'] = np.mean(zero_crossing_rate)
        features['rms_energy'] = np.mean(rms)
        
        # MFCC statistics
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        features['mfcc_1'] = np.mean(mfccs[1])
        features['mfcc_2'] = np.mean(mfccs[2])
        features['mfcc_3'] = np.mean(mfccs[3])
        
        # Chroma statistics
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        features['chroma_max'] = np.max(chroma)
        
        # Spectral contrast
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)
        
        # Tonnetz
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        
        return features
    
    def _classify_from_librosa_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """
        Classify instruments based on librosa features.
        Uses research-based heuristics for better accuracy.
        """
        instrument_scores = {}
        
        # Normalize features
        sc = min(features['spectral_centroid'] / 4000.0, 1.0)
        sr = min(features['spectral_rolloff'] / 8000.0, 1.0)
        sb = min(features['spectral_bandwidth'] / 2000.0, 1.0)
        zcr = min(features['zero_crossing_rate'] * 2.0, 1.0)
        rms = min(features['rms_energy'] * 10.0, 1.0)
        mfcc_mean = features['mfcc_mean']
        mfcc_std = features['mfcc_std']
        chroma_std = min(features['chroma_std'] * 5.0, 1.0)
        contrast_mean = min(features['spectral_contrast_mean'] * 2.0, 1.0)
        
        # Piano: Rich harmonics, moderate centroid, high contrast
        piano_score = (
            (sr * 0.4) +  # High rolloff (rich harmonics)
            ((1 - abs(sc - 0.5)) * 0.3) +  # Moderate centroid
            (contrast_mean * 0.2) +  # High spectral contrast
            (chroma_std * 0.1)  # Some pitch variance
        )
        instrument_scores["Piano"] = piano_score
        
        # Guitar: Higher centroid, good harmonics, moderate transients
        guitar_score = (
            (sc * 0.4) +  # Higher centroid
            (sr * 0.3) +  # Good harmonics
            (zcr * 0.2) +  # Moderate transients
            (rms * 0.1)  # Moderate energy
        )
        instrument_scores["Guitar"] = guitar_score
        
        # Violin: Very high centroid, rich harmonics, high contrast
        violin_score = (
            (sc * 0.5) +  # Very high centroid
            (sr * 0.3) +  # Rich harmonics
            (contrast_mean * 0.2)  # High contrast
        )
        instrument_scores["Violin"] = violin_score
        
        # Drums: Very high ZCR, high centroid, low bandwidth
        drums_score = (
            (zcr * 0.6) +  # Very high transients
            (sc * 0.3) +  # High centroid (cymbals)
            ((1 - sb) * 0.1)  # Low bandwidth (noise-like)
        )
        instrument_scores["Drums"] = drums_score
        
        # Bass: Low centroid, low rolloff, high energy
        bass_score = (
            ((1 - sc) * 0.5) +  # Low centroid
            ((1 - sr) * 0.3) +  # Low rolloff
            (rms * 0.2)  # High energy
        )
        instrument_scores["Bass"] = bass_score
        
        # Vocals: Very strict detection to avoid false positives
        # Vocals should have specific characteristics: moderate centroid, good formants, reasonable complexity
        vocals_base_score = (
            ((1 - abs(sc - 0.4)) * 0.3) +  # Moderate centroid (voice range)
            (chroma_std * 0.4) +  # Good pitch variance (formants)
            (rms * 0.2) +  # Moderate energy
            (mfcc_std * 0.1)  # Some harmonic complexity
        )
        
        # Apply strict penalties for non-vocal characteristics
        vocals_score = vocals_base_score
        
        # Heavy penalty if spectral contrast is too low (indicates simple tones)
        if features['spectral_contrast_mean'] < 0.5:
            vocals_score *= 0.2
            
        # Heavy penalty if bandwidth is too narrow (indicates simple tones)
        if features['spectral_bandwidth'] < 1200:
            vocals_score *= 0.3
            
        # Heavy penalty if spectral rolloff is too high (indicates instruments)
        if sr > 0.8:
            vocals_score *= 0.4
            
        # Penalty if zero crossing rate is too high (indicates transients/instruments)
        if zcr > 0.3:
            vocals_score *= 0.6
            
        instrument_scores["Vocals"] = vocals_score
        
        # Check if this is a simple tone (low spectral complexity)
        # Simple tones have low spectral contrast, low bandwidth, and high chroma consistency
        is_simple_tone = (
            features['spectral_contrast_mean'] < 0.3 or 
            features['spectral_bandwidth'] < 800 or
            chroma_std < 0.1  # Very consistent pitch
        )
        
        if is_simple_tone:
            # Very simple tone - significantly reduce all scores
            for instrument in instrument_scores:
                instrument_scores[instrument] *= 0.2  # Heavy penalty for simple tones
            instrument_scores["Unknown"] = 0.9  # Simple tones are likely unknown
        
        # Additional check: if all scores are very low, classify as unknown
        max_score_before_norm = max(instrument_scores.values()) if instrument_scores.values() else 0
        if max_score_before_norm < 0.3:  # Very low confidence in all instruments
            for instrument in instrument_scores:
                instrument_scores[instrument] = 0.1
            instrument_scores["Unknown"] = 0.9
        
        # Don't normalize - use absolute scores with threshold
        # This prevents one instrument from always being 1.0
        for instrument in instrument_scores:
            instrument_scores[instrument] = min(instrument_scores[instrument], 1.0)
        
        return instrument_scores
    
    def get_top_instruments(self, audio_data: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """
        Get top K most likely instruments.
        
        Args:
            audio_data: Audio data as numpy array
            top_k: Number of top instruments to return
            
        Returns:
            List of (instrument_name, confidence) tuples
        """
        scores = self.classify_audio(audio_data)
        
        # Sort by confidence and return top K
        sorted_instruments = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_instruments[:top_k]
    
    def get_dominant_instrument(self, audio_data: np.ndarray, threshold: float = 0.3) -> Tuple[str, float]:
        """
        Get the dominant instrument if confidence is above threshold.
        
        Args:
            audio_data: Audio data as numpy array
            threshold: Minimum confidence threshold
            
        Returns:
            Tuple of (instrument_name, confidence)
        """
        scores = self.classify_audio(audio_data)
        
        if not scores:
            return ("Unknown", 0.0)
        
        # Get the instrument with highest confidence
        best_instrument = max(scores.items(), key=lambda x: x[1])
        
        # Only return if above threshold
        if best_instrument[1] >= threshold:
            return best_instrument
        else:
            return ("Unknown", best_instrument[1])
