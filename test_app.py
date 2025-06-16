#!/usr/bin/env python3
"""
Test script for MusicBar core functionality.
This tests audio capture and monitoring without the GUI.
"""

import time
import sys
from src.audio.capture import AudioCapture, AudioCaptureError
from src.audio.monitor import AudioMonitor
from src.utils.config import config

def test_audio_devices():
    """Test audio device detection."""
    print("🎧 Testing audio device detection...")
    devices = AudioCapture.list_devices()
    
    if not devices:
        print("❌ No audio devices found!")
        return False
    
    print(f"✅ Found {len(devices)} audio devices:")
    for device in devices:
        print(f"  - {device['id']}: {device['name']} ({device['channels']}ch @ {device['sample_rate']}Hz)")
    
    return True

def test_audio_capture():
    """Test basic audio capture functionality."""
    print("\n🎵 Testing audio capture...")
    
    # Set up monitoring
    metrics_received = 0
    
    def on_metrics_update(metrics):
        nonlocal metrics_received
        metrics_received += 1
        if metrics_received <= 3:  # Print first few updates
            print(f"  📊 Metrics {metrics_received}: Volume={metrics.overall_volume:.4f}, "
                  f"Balance={metrics.stereo_balance:.2f}, Peak={metrics.peak_frequency:.0f}Hz")
    
    monitor = AudioMonitor(update_callback=on_metrics_update)
    capture = AudioCapture(callback=monitor.process_audio)
    
    try:
        print("  🚀 Starting audio capture...")
        capture.start()
        
        print("  ⏱️ Monitoring for 5 seconds...")
        start_time = time.time()
        while time.time() - start_time < 5.0:
            time.sleep(0.1)
        
        print(f"  ✅ Received {metrics_received} metric updates")
        
        # Get final metrics
        final_metrics = monitor.get_metrics()
        volume_db = monitor.get_volume_db()
        stereo_info = monitor.get_stereo_info()
        
        print(f"  📈 Final metrics:")
        print(f"    Volume: {volume_db['overall_db']:.1f}dB")
        print(f"    Stereo: {stereo_info['balance_description']} ({stereo_info['width_description']})")
        print(f"    Peak Frequency: {final_metrics.peak_frequency:.0f}Hz")
        
        return metrics_received > 0
        
    except AudioCaptureError as e:
        print(f"  ❌ Audio capture error: {e}")
        return False
    finally:
        capture.stop()
        print("  🛑 Audio capture stopped")

def test_configuration():
    """Test configuration system."""
    print("\n⚙️ Testing configuration...")
    
    print(f"  📝 App: {config.app_name} v{config.version}")
    print(f"  🎚️ Audio: {config.audio.sample_rate}Hz, {config.audio.channels}ch, buffer={config.audio.chunk_size}")
    print(f"  🖥️ UI: {config.ui.update_interval}s updates, history={config.ui.volume_history_length}")
    print(f"  🐛 Debug: {config.debug}")
    
    return True

def main():
    """Run all tests."""
    print("🧪 MusicBar Core Functionality Test")
    print("=" * 40)
    
    # Enable debug for testing
    config.debug = True
    
    tests = [
        ("Configuration", test_configuration),
        ("Audio Devices", test_audio_devices),
        ("Audio Capture", test_audio_capture),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            print(f"\n📋 Running: {test_name}")
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! MusicBar core functionality is working.")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 