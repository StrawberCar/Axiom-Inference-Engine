"""
axim - Axiom Model Package Format

Load and inspect .axim files. That's about it.
"""

from .core import load_axim, write_axim, print_axim_info, AXIMHeader, SectionType

__all__ = ["load_axim", "write_axim", "print_axim_info", "AXIMHeader", "SectionType"]
