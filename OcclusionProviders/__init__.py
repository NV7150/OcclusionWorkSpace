"""
OcclusionProviders module for the Occlusion Framework.

This module provides classes for generating occlusion masks from frame data.
"""

from .SimpleOcclusionProvider import SimpleOcclusionProvider
from .DepthThresholdOcclusionProvider import (
    DepthThresholdOcclusionProvider,
    DepthGradientOcclusionProvider,
    HybridOcclusionProvider
)

__all__ = [
    'SimpleOcclusionProvider',
    'DepthThresholdOcclusionProvider',
    'DepthGradientOcclusionProvider',
    'HybridOcclusionProvider'
]