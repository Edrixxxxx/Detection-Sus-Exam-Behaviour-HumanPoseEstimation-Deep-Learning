"""
AI Examination Behavior Proctoring System - Main Desktop Launcher
==================================================================
Root launcher script. Starts the modern PyQt6 Graphical User Interface.

Usage:
    python main.py
"""

import sys
from pathlib import Path

# Add 1_Source_Code to sys.path
source_dir = Path(__file__).resolve().parent / "1_Source_Code"
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from app_gui import main

if __name__ == "__main__":
    main()
