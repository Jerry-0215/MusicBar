#!/usr/bin/env python3
"""
MusicBar - Real-time Audio Analysis and Visualization
Main entry point for the application.
"""
import sys
import os
from .gui.menu_bar import run_app
from .utils.config import config

def main():
    """Main entry point for MusicBar application."""
    print(f"Starting {config.app_name} v{config.version}")
    
    # Enable debug mode if requested
    if "--debug" in sys.argv:
        config.debug = True
        print("Debug mode enabled")
    
    # Check for help flag
    if "--help" in sys.argv or "-h" in sys.argv:
        print_help()
        return
    
    # Check macOS version (rumps requires macOS)
    if sys.platform != "darwin":
        print("Error: MusicBar currently only supports macOS")
        sys.exit(1)
    
    try:
        # Run the menu bar app
        run_app()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error: {e}")
        if config.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def print_help():
    """Print help information."""
    help_text = f"""
{config.app_name} v{config.version}
Real-time Audio Monitoring and Visualization for macOS

USAGE:
    python -m src.main [OPTIONS]

OPTIONS:
    --debug     Enable debug output
    --help, -h  Show this help message

DESCRIPTION:
    MusicBar is a macOS menu bar application that provides real-time
    monitoring and analysis of system audio. It displays:
    
    • Volume levels in dB
    • Stereo field information (balance and width)
    • Peak frequency analysis
    • Real-time audio visualization
    
FEATURES:
    • Real-time audio capture from system devices
    • Stereo field analysis and visualization
    • Volume monitoring with dB readings
    • Frequency domain analysis
    • Device selection and switching
    • Low-latency processing

REQUIREMENTS:
    • macOS 10.14 or later
    • Python 3.8 or later
    • Audio input device (microphone or system audio)

PERMISSIONS:
    The app may request microphone or system audio access permissions.
    Grant these to enable audio monitoring functionality.

For more information, visit: https://github.com/your-repo/musicbar
"""
    print(help_text)

if __name__ == "__main__":
    main() 