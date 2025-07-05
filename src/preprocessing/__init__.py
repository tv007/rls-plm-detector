"""
Preprocessing module for WASM 2019 leg movement detection software.

This module implements STEP 1 of the WASM 2019 algorithm:
- EMG signal filtering (10-100 Hz bandpass)
- Signal rectification
- Moving average smoothing (300ms window)
"""

from .signal_processing import (
    preprocess_emg,
    apply_bandpass_filter,
    rectify_signal,
    smooth_signal,
    assess_signal_quality
)

__all__ = [
    'preprocess_emg',
    'apply_bandpass_filter',
    'rectify_signal',
    'smooth_signal',
    'assess_signal_quality'
] 