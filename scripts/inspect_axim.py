"""
Inspect an .axim file.

Usage:
    python inspect_axim.py model.axim
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from axim.core import print_axim_info

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python inspect_axim.py <file.axim>")
        sys.exit(1)
    print_axim_info(sys.argv[1])
