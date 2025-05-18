"""
DataLoaders module for the Occlusion Framework.

This module provides classes for loading frame data from various sources.
"""

from .Frame import Frame
from .BaseFrameLoader import BaseFrameLoader
from .UniformedFrameLoader import UniformedFrameLoader
from .SeparatedFrameLoader import SeparatedFrameLoader

__all__ = [
    'Frame',
    'BaseFrameLoader',
    'UniformedFrameLoader',
    'SeparatedFrameLoader'
]