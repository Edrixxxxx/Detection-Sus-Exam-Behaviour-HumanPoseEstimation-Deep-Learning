"""
Application Entry Point - Desktop Proctoring Surveillance System
================================================================
Main entry point for starting the PyQt6 Graphical User Interface.
"""

import sys
from pathlib import Path

# Add 1_Source_Code to sys.path
source_dir = Path(__file__).resolve().parent.parent
if str(source_dir) not in sys.path:
    sys.path.insert(0, str(source_dir))

from app_gui import main

if __name__ == "__main__":
    main()
