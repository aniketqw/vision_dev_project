"""
Training Analytics Package

A comprehensive logging and debugging callback for PyTorch Lightning training.
Captures detailed metrics including gradients, class-wise performance, batch metrics, 
and misclassified images in base64 format.
"""

from .debug_logger import DebugLogger

__version__ = "1.0.0"
__author__ = "Vision Dev Project"

__all__ = ['DebugLogger']
