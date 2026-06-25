"""
Inspect an .axim file.

Usage:
    python inspect.py model.axim
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from axim.core import print_axim_info

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python inspect.py <file.axim>")
        sys.exit(1)
    print_axim_info(sys.argv[1])
