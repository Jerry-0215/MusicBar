# MusicBar 

Real-time audio monitoring and visualization for macOS menu bar.

**Features:**
- **Real-time volume monitoring with dB display**: `numpy.sqrt()` + `20 * numpy.log10()` for RMS calculation and dB conversion
- **BPM detection**: `essentia.standard.RhythmExtractor2013(method="multifeature")` from Essentia library
- **Chord recognition with confidence scoring**: `essentia.standard.SpectralPeaks()` + `essentia.standard.HPCP()` + custom chord templates with `numpy.dot()` correlation matching
- **Multi-instrument detection**: `librosa.feature.*` functions (spectral_centroid, mfcc, chroma_stft, spectral_contrast) + research-based heuristics in `EnhancedInstrumentClassifier`
- **Stereo image visualization**: Custom `StereoImageView` using `AppKit.NSBezierPath` for circular visualization with `numpy.cos()` and `numpy.sin()` for positioning
- **Dynamic audio device selection**: `sounddevice.query_devices()` and `sounddevice.InputStream()` for device enumeration and switching
- **Native macOS integration**: `AppKit.NSStatusBar`, `AppKit.NSMenu`, `PyObjC` for menu bar interface and system permissions

## Detection Functions & Libraries Used

**Core Frameworks:**
- **AppKit** - Native macOS GUI framework 
- **PyObjC** - Python-Objective-C bridge for system integration
- **sounddevice** - Real-time audio capture and streaming
- **Essentia** - Professional audio analysis library

**Audio Processing:**
- **librosa** - Advanced audio feature extraction
- **numpy** - Numerical operations and array processing
- **scikit-learn** - Machine learning support for librosa

**System Integration:**
- **pyobjc-framework-Cocoa** - Core macOS integration
- **pyobjc-framework-AVFoundation** - Audio system access
- **pyobjc-framework-Quartz** - Screen recording permissions

**Analysis Capabilities:**
- Volume/RMS analysis with dB conversion
- Spectral analysis (centroid, rolloff, bandwidth)
- MFCC feature extraction
- Chroma-based chord detection
- Rhythm extraction for BPM
- Enhanced instrument classification with research-based profiles

## Limitations and Future Prospects

**Current Limitations:**
- Requires blackhole (cannot monitor system audio directly)
- Instrument detection (tested a few approaches but none work well)
- Chord detection is limited in accuracy, variety and speed

**Future Enhancements:**
- System audio monitoring (requires additional macOS permissions)
- Song recognition using acoustic fingerprinting (pyacoustid)
- Extended chord vocabulary (7th chords, extensions)
- Machine learning model improvement with instrument detection
- Soundfield visualization

