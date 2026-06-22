# Copyright (c) 2025 Daniele De Sensi e Saverio Pasqualoni
# Licensed under the MIT License

"""
Plotting utilities for the pico project.

This package exposes individual plotting helpers (one per module) and a CLI
entrypoint that orchestrates them.  Import specific ``generate_*`` functions
from :mod:`plot.plots` or invoke ``python -m plot`` to use the command line
interface.
"""

from .cli import main

__all__ = ["main"]
