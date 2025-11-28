"""
YAMNet-based instrument classifier using TensorFlow Hub.
This provides professional-grade instrument detection using Google's pre-trained model.
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..utils.config import config

class YAMNetClassifier:
    """
    Professional instrument classification using Google's YAMNet model.
    YAMNet can detect 521 different audio events including musical instruments.
    """
    
    def __init__(self):
        self.sample_rate = config.audio.sample_rate
        self.model = None
        self.class_names = None
        
        # Map YAMNet classes to our instrument categories
        self.instrument_mapping = {
            # Strings
            'Guitar': ['Guitar', 'Acoustic guitar', 'Electric guitar'],
            'Violin': ['Violin', 'Fiddle'],
            'Bass': ['Bass guitar', 'Double bass', 'String bass'],
            'Cello': ['Cello'],
            'Harp': ['Harp'],
            
            # Keyboards
            'Piano': ['Piano', 'Harpsichord', 'Keyboard'],
            'Organ': ['Organ', 'Pipe organ', 'Hammond organ'],
            'Synthesizer': ['Synthesizer', 'Synthesizer music'],
            
            # Wind instruments
            'Saxophone': ['Saxophone', 'Sax', 'Tenor saxophone', 'Alto saxophone'],
            'Trumpet': ['Trumpet'],
            'Trombone': ['Trombone'],
            'Tuba': ['Tuba'],
            'Flute': ['Flute', 'Recorder'],
            'Clarinet': ['Clarinet'],
            'Oboe': ['Oboe'],
            'Harmonica': ['Harmonica', 'Mouth organ'],
            
            # Percussion
            'Drums': ['Drum', 'Drum kit', 'Snare drum', 'Bass drum', 'Cymbal'],
            'Xylophone': ['Xylophone', 'Marimba', 'Vibraphone'],
            
            # Other
            'Accordion': ['Accordion'],
            'Banjo': ['Banjo'],
            'Vocals': ['Singing', 'Choir', 'Chorus', 'Vocal music']
        }
        
        if config.debug:
            print("YAMNetClassifier initialized")
    
    def _load_model(self):
        """Lazy load the YAMNet model to avoid startup delays."""
        if self.model is None:
            try:
                import tensorflow as tf
                import tensorflow_hub as hub
                
                # Load YAMNet model
                self.model = hub.load('https://tfhub.dev/google/yamnet/1')
                self.class_names = self.model.class_names.numpy()
                
                if config.debug:
                    print("YAMNet model loaded successfully")
                    
            except Exception as e:
                if config.debug:
                    print(f"Failed to load YAMNet model: {e}")
                self.model = None
                self.class_names = None
    
    def classify_audio(self, audio_data: np.ndarray) -> Dict[str, float]:
        """
        Classify instruments in audio data using YAMNet.
        
        Args:
            audio_data: Audio data as numpy array
            
        Returns:
            Dictionary of instrument names and confidence scores
        """
        try:
            # Load model if not already loaded
            self._load_model()
            
            if self.model is None:
                return {"Unknown": 0.0}
            
            # Convert stereo to mono if needed
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Ensure audio is float32 and in the right range
            audio_data = audio_data.astype(np.float32)
            
            # YAMNet expects 16kHz audio, so resample if needed
            if self.sample_rate != 16000:
                audio_data = self._resample_audio(audio_data, self.sample_rate, 16000)
            
            # Run YAMNet inference
            scores, embeddings, spectrogram = self.model(audio_data)
            
            # Get top predictions
            top_scores = np.argsort(scores.numpy().mean(axis=0))[-10:][::-1]  # Top 10
            
            # Map YAMNet predictions to our instrument categories
            instrument_scores = {}
            
            for class_idx in top_scores:
                class_name = self.class_names[class_idx].decode('utf-8')
                confidence = float(scores.numpy().mean(axis=0)[class_idx])
                
                # Map to our instrument categories
                mapped_instrument = self._map_to_instrument(class_name)
                if mapped_instrument:
                    # Take the maximum confidence for each instrument category
                    if mapped_instrument in instrument_scores:
                        instrument_scores[mapped_instrument] = max(
                            instrument_scores[mapped_instrument], confidence
                        )
                    else:
                        instrument_scores[mapped_instrument] = confidence
            
            # Add "Unknown" if no instruments detected with high confidence
            max_confidence = max(instrument_scores.values()) if instrument_scores else 0.0
            if max_confidence < 0.3:
                instrument_scores["Unknown"] = 0.8
            
            return instrument_scores
            
        except Exception as e:
            if config.debug:
                print(f"YAMNet classification failed: {e}")
            return {"Unknown": 0.0}
    
    def _map_to_instrument(self, yamnet_class: str) -> Optional[str]:
        """Map YAMNet class name to our instrument categories."""
        yamnet_class_lower = yamnet_class.lower()
        
        for instrument, yamnet_classes in self.instrument_mapping.items():
            for yamnet_class_name in yamnet_classes:
                if yamnet_class_name.lower() in yamnet_class_lower:
                    return instrument
        
        return None
    
    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Simple resampling using numpy (not ideal but works for basic cases)."""
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            # Fallback: simple decimation/interpolation
            ratio = target_sr / orig_sr
            if ratio < 1:
                # Decimation
                step = int(1 / ratio)
                return audio_data[::step]
            else:
                # Interpolation (linear)
                old_indices = np.arange(len(audio_data))
                new_length = int(len(audio_data) * ratio)
                new_indices = np.linspace(0, len(audio_data) - 1, new_length)
                return np.interp(new_indices, old_indices, audio_data)
    
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
