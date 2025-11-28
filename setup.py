#!/usr/bin/env python3
"""
Setup script for MusicBar - Real-time Audio Analysis and Visualization
"""
from setuptools import setup, find_packages

APP = ['src/main.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': True,
    # 'iconfile': 'icon.icns',  # You'll need to create an icon file for your app
    'packages': ['numpy', 'essentia', 'sounddevice', '_sounddevice_data', 'six', 'src'],
    'excludes': ['numpy', 'essentia'],
    'plist': {
        'CFBundleName': 'MusicBar',
        'CFBundleDisplayName': 'MusicBar',
        'CFBundleGetInfoString': "Real-time music information in your menu bar",
        'CFBundleIdentifier': "com.yourdomain.musicbar",
        'CFBundleVersion': "0.1.0",
        'CFBundleShortVersionString': "0.1.0",
        'NSHumanReadableCopyright': u"Copyright © 2024, Your Name, All Rights Reserved",
        'NSPrincipalClass': 'NSApplication',
        'NSHighResolutionCapable': 'True',
        'LSUIElement': 'True',  # Hides the dock icon
        'NSMicrophoneUsageDescription': 'MusicBar needs access to the microphone to analyze audio in real time.',
    },
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
    packages=find_packages(),
) 