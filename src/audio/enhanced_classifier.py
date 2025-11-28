"""
Enhanced instrument classifier combining multiple approaches.
Uses improved librosa features with better heuristics and confidence scoring.
"""
import numpy as np
import librosa
from typing import Dict, List, Tuple, Optional
from ..utils.config import config

class EnhancedInstrumentClassifier:
    """
    Enhanced instrument classification using improved librosa features
    with better heuristics and confidence scoring.
    
    This approach combines:
    1. Advanced librosa feature extraction
    2. Research-based instrument characteristics
    3. Improved confidence scoring
    4. Better handling of edge cases
    """
    
    def __init__(self):
        self.sample_rate = config.audio.sample_rate
        
        # Instrument characteristics based on music research
        self.instrument_profiles = {
            'Piano': {
                'spectral_centroid_range': (500, 3000),
                'spectral_rolloff_range': (2000, 8000),
                'spectral_bandwidth_range': (800, 2000),
                'zero_crossing_rate_range': (0.01, 0.15),
                'rms_energy_range': (0.1, 1.0),
                'mfcc_variance_range': (50, 200),
                'chroma_consistency_range': (0.3, 0.8),
                'spectral_contrast_range': (0.3, 0.8),
                'tempo_range': (60, 200),
                'harmonic_richness': 0.8
            },
            'Guitar': {
                'spectral_centroid_range': (800, 4000),
                'spectral_rolloff_range': (3000, 10000),
                'spectral_bandwidth_range': (1000, 2500),
                'zero_crossing_rate_range': (0.02, 0.25),
                'rms_energy_range': (0.2, 1.0),
                'mfcc_variance_range': (60, 250),
                'chroma_consistency_range': (0.4, 0.9),
                'spectral_contrast_range': (0.4, 0.9),
                'tempo_range': (80, 180),
                'harmonic_richness': 0.9
            },
            'Violin': {
                'spectral_centroid_range': (1500, 5000),
                'spectral_rolloff_range': (4000, 12000),
                'spectral_bandwidth_range': (1200, 3000),
                'zero_crossing_rate_range': (0.01, 0.2),
                'rms_energy_range': (0.15, 0.9),
                'mfcc_variance_range': (70, 300),
                'chroma_consistency_range': (0.5, 0.95),
                'spectral_contrast_range': (0.5, 0.95),
                'tempo_range': (60, 220),
                'harmonic_richness': 0.95
            },
            'Drums': {
                'spectral_centroid_range': (1000, 6000),
                'spectral_rolloff_range': (2000, 8000),
                'spectral_bandwidth_range': (1500, 4000),
                'zero_crossing_rate_range': (0.1, 0.8),
                'rms_energy_range': (0.3, 1.0),
                'mfcc_variance_range': (100, 400),
                'chroma_consistency_range': (0.1, 0.6),
                'spectral_contrast_range': (0.2, 0.7),
                'tempo_range': (80, 200),
                'harmonic_richness': 0.3
            },
            'Bass': {
                'spectral_centroid_range': (100, 1500),
                'spectral_rolloff_range': (500, 3000),
                'spectral_bandwidth_range': (400, 1200),
                'zero_crossing_rate_range': (0.005, 0.1),
                'rms_energy_range': (0.4, 1.0),
                'mfcc_variance_range': (30, 150),
                'chroma_consistency_range': (0.6, 0.95),
                'spectral_contrast_range': (0.2, 0.6),
                'tempo_range': (60, 160),
                'harmonic_richness': 0.7
            },
            'Vocals': {
                'spectral_centroid_range': (800, 2500),
                'spectral_rolloff_range': (2000, 6000),
                'spectral_bandwidth_range': (600, 1800),
                'zero_crossing_rate_range': (0.01, 0.2),
                'rms_energy_range': (0.2, 0.8),
                'mfcc_variance_range': (40, 200),
                'chroma_consistency_range': (0.7, 0.95),
                'spectral_contrast_range': (0.3, 0.7),
                'tempo_range': (60, 180),
                'harmonic_richness': 0.6
            }
        }
        
        if config.debug:
            print("EnhancedInstrumentClassifier initialized with research-based profiles")
    
    def classify_audio(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Classify instruments in audio data using enhanced features.
        
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
            
            # Extract comprehensive features
            features = self._extract_enhanced_features(audio_data)
            
            # Classify using enhanced profiles
            instrument_scores = self._classify_with_profiles(features)
            
            # Apply confidence adjustments
            instrument_scores = self._apply_confidence_adjustments(instrument_scores, features)
            
            return instrument_scores
            
        except Exception as e:
            if config.debug:
                print(f"Enhanced classification failed: {e}")
            return {"Unknown": 0.0}
    
    def _extract_enhanced_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract comprehensive audio features using librosa."""
        features = {}
        
        # For very short audio, use simple features instead of librosa
        if len(audio_data) < 512:  # Less than ~12ms at 44.1kHz
            return self._extract_simple_features(audio_data)
        
        # Use appropriate FFT size for the audio length
        n_fft = min(1024, len(audio_data) // 2)
        hop_length = n_fft // 4
        
        # Basic spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_data, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_data, hop_length=hop_length)[0]
        rms_energy = librosa.feature.rms(y=audio_data, hop_length=hop_length)[0]
        
        # MFCC features
        mfccs = librosa.feature.mfcc(y=audio_data, sr=self.sample_rate, n_mfcc=13, n_fft=n_fft, hop_length=hop_length)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=audio_data, sr=self.sample_rate)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=self.sample_rate, n_fft=n_fft, hop_length=hop_length)
        
        # Tonnetz
        tonnetz = librosa.feature.tonnetz(y=audio_data, sr=self.sample_rate)
        
        # Tempo (only if audio is long enough)
        if len(audio_data) > self.sample_rate:  # At least 1 second
            try:
                tempo, _ = librosa.beat.beat_track(y=audio_data, sr=self.sample_rate)
            except:
                tempo = 0.0
        else:
            tempo = 0.0
        
        # Polyphonic features
        poly_features = librosa.feature.poly_features(y=audio_data, sr=self.sample_rate)
        
        # Average features across time
        features['spectral_centroid'] = np.mean(spectral_centroids)
        features['spectral_rolloff'] = np.mean(spectral_rolloff)
        features['spectral_bandwidth'] = np.mean(spectral_bandwidth)
        features['zero_crossing_rate'] = np.mean(zero_crossing_rate)
        features['rms_energy'] = np.mean(rms_energy)
        features['tempo'] = tempo
        
        # MFCC statistics
        features['mfcc_mean'] = np.mean(mfccs)
        features['mfcc_std'] = np.std(mfccs)
        features['mfcc_variance'] = np.var(mfccs)
        
        # Chroma statistics
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        features['chroma_consistency'] = 1.0 - np.std(chroma) / (np.mean(chroma) + 1e-8)
        
        # Spectral contrast
        features['spectral_contrast_mean'] = np.mean(spectral_contrast)
        features['spectral_contrast_std'] = np.std(spectral_contrast)
        
        # Tonnetz
        features['tonnetz_mean'] = np.mean(tonnetz)
        features['tonnetz_std'] = np.std(tonnetz)
        
        # Polyphonic features
        features['poly_features_mean'] = np.mean(poly_features)
        features['poly_features_std'] = np.std(poly_features)
        
        # Harmonic/percussive separation (only if audio is long enough)
        if len(audio_data) > self.sample_rate:  # At least 1 second
            try:
                y_harmonic, y_percussive = librosa.effects.hpss(audio_data)
                features['harmonic_ratio'] = np.sum(y_harmonic**2) / (np.sum(y_harmonic**2) + np.sum(y_percussive**2) + 1e-8)
            except:
                features['harmonic_ratio'] = 0.5  # Default value
        else:
            features['harmonic_ratio'] = 0.5  # Default value
        
        return features
    
    def _extract_simple_features(self, audio_data: np.ndarray) -> Dict[str, float]:
        """Extract simple features for very short audio segments."""
        features = {}
        
        # Basic statistics
        features['spectral_centroid'] = np.mean(np.abs(audio_data)) * 2000  # Rough approximation
        features['spectral_rolloff'] = np.percentile(np.abs(audio_data), 85) * 4000
        features['spectral_bandwidth'] = np.std(audio_data) * 1000
        features['zero_crossing_rate'] = np.mean(np.diff(np.sign(audio_data)) != 0)
        features['rms_energy'] = np.sqrt(np.mean(audio_data**2))
        features['mfcc_variance'] = np.var(audio_data) * 10000
        features['chroma_consistency'] = 0.5  # Default value
        features['spectral_contrast_mean'] = np.std(audio_data) * 10  # Rough approximation
        
        return features
    
    def _classify_with_profiles(self, features: Dict[str, float]) -> Dict[str, float]:
        """Classify using simplified heuristics based on audio characteristics."""
        instrument_scores = {}
        
        # Extract key features
        spectral_centroid = features.get('spectral_centroid', 0)
        spectral_rolloff = features.get('spectral_rolloff', 0)
        zero_crossing_rate = features.get('zero_crossing_rate', 0)
        rms_energy = features.get('rms_energy', 0)
        mfcc_variance = features.get('mfcc_variance', 0)
        chroma_consistency = features.get('chroma_consistency', 0)
        
        # Piano: Moderate centroid, high rolloff, good harmonics
        piano_score = 0.0
        if 500 <= spectral_centroid <= 3000:
            piano_score += 0.4
        if spectral_rolloff > 2000:
            piano_score += 0.3
        if mfcc_variance > 1000:
            piano_score += 0.3
        instrument_scores["Piano"] = piano_score
        
        # Guitar: Higher centroid, good harmonics
        guitar_score = 0.0
        if 800 <= spectral_centroid <= 4000:
            guitar_score += 0.5
        if spectral_rolloff > 3000:
            guitar_score += 0.3
        if zero_crossing_rate > 0.02:
            guitar_score += 0.2
        instrument_scores["Guitar"] = guitar_score
        
        # Violin: Very high centroid, rich harmonics
        violin_score = 0.0
        if spectral_centroid > 1500:
            violin_score += 0.6
        if spectral_rolloff > 4000:
            violin_score += 0.4
        instrument_scores["Violin"] = violin_score
        
        # Drums: High ZCR, high centroid
        drums_score = 0.0
        if zero_crossing_rate > 0.1:
            drums_score += 0.6
        if spectral_centroid > 1000:
            drums_score += 0.4
        instrument_scores["Drums"] = drums_score
        
        # Bass: Low centroid, high energy
        bass_score = 0.0
        if spectral_centroid < 1500:
            bass_score += 0.6
        if rms_energy > 0.3:
            bass_score += 0.4
        instrument_scores["Bass"] = bass_score
        
        # Vocals: Moderate centroid, good pitch consistency
        vocals_score = 0.0
        if 800 <= spectral_centroid <= 2500:
            vocals_score += 0.4
        if chroma_consistency > 0.7:
            vocals_score += 0.4
        if rms_energy > 0.2:
            vocals_score += 0.2
        instrument_scores["Vocals"] = vocals_score
        
        return instrument_scores
    
    def _get_feature_weight(self, feature_name: str) -> float:
        """Get importance weight for each feature."""
        weights = {
            'spectral_centroid': 0.15,
            'spectral_rolloff': 0.12,
            'spectral_bandwidth': 0.10,
            'zero_crossing_rate': 0.15,
            'rms_energy': 0.08,
            'mfcc_variance': 0.12,
            'chroma_consistency': 0.10,
            'spectral_contrast': 0.10,
            'tempo': 0.08
        }
        return weights.get(feature_name, 0.05)
    
    def _apply_confidence_adjustments(self, scores: Dict[str, float], features: Dict[str, float]) -> Dict[str, float]:
        """Apply confidence adjustments based on audio characteristics."""
        adjusted_scores = scores.copy()
        
        # Check for simple tones (low complexity)
        complexity = (
            features.get('spectral_contrast_mean', 0) +
            features.get('mfcc_variance', 0) / 1000.0 +
            features.get('chroma_std', 0) * 10
        ) / 3.0
        
        # Debug: print complexity for troubleshooting
        if config.debug:
            print(f"[Instrument Debug] Complexity: {complexity:.3f}")
        
        # Make the complexity threshold much less strict
        if complexity < 0.05:  # Only for very simple tones
            # Very simple tone - reduce all scores and add Unknown
            for instrument in adjusted_scores:
                adjusted_scores[instrument] *= 0.3
            adjusted_scores["Unknown"] = 0.8
        else:
            # For normal audio, normalize scores to realistic ranges
            max_score = max(adjusted_scores.values()) if adjusted_scores.values() else 0.0
            
            # Scale scores to be more realistic (0.1 to 0.8 range)
            if max_score > 0:
                scale_factor = 0.6 / max_score  # Scale to max 0.6
                for instrument in adjusted_scores:
                    adjusted_scores[instrument] = min(adjusted_scores[instrument] * scale_factor, 0.8)
                
                # Only add Unknown if no instrument has reasonable confidence
                if max(adjusted_scores.values()) < 0.3:
                    adjusted_scores["Unknown"] = 0.4
        
        # Check for silence or very low energy
        if features.get('rms_energy', 0) < 0.01:
            adjusted_scores = {"Unknown": 0.9}
        
        # Check for noise (high zero crossing rate)
        if features.get('zero_crossing_rate', 0) > 0.5:
            # Likely noise or percussion
            if 'Drums' in adjusted_scores:
                adjusted_scores['Drums'] = min(adjusted_scores['Drums'] * 1.5, 1.0)
        
        return adjusted_scores
    
    def get_top_instruments(self, audio_data: np.ndarray, top_k: int = 3) -> List[Tuple[str, float]]:
        """Get top K most likely instruments."""
        scores = self.classify_audio(audio_data)
        sorted_instruments = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_instruments[:top_k]
    
    def get_dominant_instrument(self, audio_data: np.ndarray, threshold: float = 0.3) -> Tuple[str, float]:
        """Get the dominant instrument if confidence is above threshold."""
        scores = self.classify_audio(audio_data)
        
        if not scores:
            return ("Unknown", 0.0)
        
        best_instrument = max(scores.items(), key=lambda x: x[1])
        
        if best_instrument[1] >= threshold:
            return best_instrument
        else:
            return ("Unknown", best_instrument[1])
