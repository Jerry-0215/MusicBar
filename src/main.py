#!/usr/bin/env python3
"""
MusicBar - Real-time Audio Analysis and Visualization
Main entry point for the application.
"""
import sys
import os
from src.gui.menu_bar import run_app

if __name__ == "__main__":
    try:
        run_app()
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        # On macOS, write exceptions to a log file on the desktop
        # for easier debugging of the packaged .app
        error_log = os.path.expanduser("~/Desktop/MusicBar_error.log")
        with open(error_log, "a") as f:
            import traceback
            f.write(f"--- MusicBar Crash Log ---\n")
            traceback.print_exc(file=f)
            f.write("\n")
        # Re-raise the exception to ensure the OS knows the app crashed
        raise e