#!/usr/bin/env python3
"""
Simple launcher for MusicBar app.
This avoids relative import issues when running from project root.
"""
import sys
import os

# Add src directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.insert(0, src_path)

# Now we can import and run the app
from gui.menu_bar import run_app

if __name__ == "__main__":
    run_app() 