#!/usr/bin/env python3
"""
Setup script for MusicBar - Real-time Audio Analysis and Visualization
"""
from setuptools import setup, find_packages
import os

# Read the README file for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ""

# Read requirements from requirements.txt
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    requirements = []
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    return requirements

setup(
    name="musicbar",
    version="0.1.0",
    author="MusicBar Team",
    author_email="team@musicbar.app",
    description="Real-time audio monitoring and visualization for macOS menu bar",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/musicbar",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Multimedia :: Sound/Audio :: Analysis",
        "Topic :: System :: Monitoring",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    entry_points={
        "console_scripts": [
            "musicbar=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    platforms=["darwin"],  # macOS only
    keywords="audio, monitoring, visualization, macos, menubar, stereo, analysis",
    project_urls={
        "Bug Reports": "https://github.com/your-repo/musicbar/issues",
        "Source": "https://github.com/your-repo/musicbar",
        "Documentation": "https://github.com/your-repo/musicbar/wiki",
    },
) 