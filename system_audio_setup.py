#!/usr/bin/env python3
"""
System Audio Setup Helper for MusicBar
This script helps you set up system audio capture on macOS.
"""

import subprocess
import sys
from src.audio.capture import AudioCapture

def check_blackhole():
    """Check if BlackHole is installed and available."""
    print("🔍 Checking for BlackHole audio driver...")
    
    devices = AudioCapture.list_devices()
    blackhole_devices = [d for d in devices if 'blackhole' in d['name'].lower()]
    
    if blackhole_devices:
        print("✅ BlackHole found!")
        for device in blackhole_devices:
            print(f"   - {device['name']} (Device ID: {device['id']})")
        return blackhole_devices
    else:
        print("❌ BlackHole not found")
        print("📥 Install BlackHole from: https://github.com/ExistentialAudio/BlackHole")
        return []

def setup_instructions():
    """Provide step-by-step setup instructions."""
    print("\n" + "="*60)
    print("🎵 SYSTEM AUDIO SETUP INSTRUCTIONS")
    print("="*60)
    
    print("\n1️⃣ OPTION A: Use BlackHole (Recommended)")
    print("   • You already have BlackHole 2ch installed! ✅")
    print("   • Set up Multi-Output Device:")
    print("     - Open Audio MIDI Setup (Cmd+Space, type 'Audio MIDI')")
    print("     - Click '+' button → 'Create Multi-Output Device'")
    print("     - Check BOTH boxes:")
    print("       ☑️ MacBook Pro扬声器 (so you can still hear)")
    print("       ☑️ BlackHole 2ch (for MusicBar to capture)")
    print("     - Set Multi-Output Device as system output")
    print("   • In MusicBar: Select 'BlackHole 2ch' as audio device")
    
    print("\n2️⃣ OPTION B: Use Screen Recording Permission")
    print("   • Grant Screen Recording permission to Terminal")
    print("   • This allows access to system audio directly")
    print("   • System Preferences → Security → Privacy → Screen Recording")
    
    print("\n3️⃣ TEST YOUR SETUP:")
    print("   • Play music from Spotify/YouTube")
    print("   • Check MusicBar menu for volume changes")
    print("   • Should show audio levels from your music!")

def test_current_setup():
    """Test what audio we're currently capturing."""
    print("\n🧪 TESTING CURRENT AUDIO CAPTURE...")
    print("Make some noise (talk, play music) and watch the levels:")
    
    from src.audio.monitor import AudioMonitor
    
    def show_audio_levels(metrics):
        volume_db = 20 * np.log10(metrics.overall_volume) if metrics.overall_volume > 0 else -60
        bar_length = max(0, min(20, int((volume_db + 60) / 3)))  # Scale -60dB to 0dB as 0-20 chars
        bar = "█" * bar_length + "░" * (20 - bar_length)
        print(f"\r🔊 {bar} {volume_db:.1f}dB", end="", flush=True)
    
    import numpy as np
    monitor = AudioMonitor(update_callback=show_audio_levels)
    capture = AudioCapture(callback=monitor.process_audio)
    
    try:
        print("🎤 Starting audio capture test (10 seconds)...")
        capture.start()
        
        import time
        time.sleep(10)
        
        print("\n✅ Test complete!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        capture.stop()

def main():
    """Main setup function."""
    print("🎵 MusicBar System Audio Setup")
    print("=" * 40)
    
    # Check BlackHole
    blackhole_devices = check_blackhole()
    
    # Show instructions
    setup_instructions()
    
    # Ask user what they want to do
    print("\n" + "="*60)
    print("WHAT WOULD YOU LIKE TO DO?")
    print("1 - Test current audio capture")
    print("2 - Show device list")
    print("3 - Just show instructions again")
    print("q - Quit")
    
    choice = input("\nEnter choice (1/2/3/q): ").strip()
    
    if choice == "1":
        test_current_setup()
    elif choice == "2":
        print("\n🎧 Available Audio Devices:")
        devices = AudioCapture.list_devices()
        for device in devices:
            print(f"  {device['id']:2d}: {device['name']} ({device['channels']}ch)")
    elif choice == "3":
        setup_instructions()
    
    print("\n🎵 MusicBar should now be running in your menu bar!")
    print("Look for the 🎵 icon in the top-right corner and click it.")

if __name__ == "__main__":
    main() 