#!/usr/bin/env python3
"""
Audio Transcriber - Main Entry Point
A local audio-to-text transcription tool powered by OpenAI Whisper.

Usage:
    python main.py          # Launch GUI
    python main.py --cli    # Command-line mode (planned)
"""

import sys
import os

# Add the app directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui import main


if __name__ == "__main__":
    main()
