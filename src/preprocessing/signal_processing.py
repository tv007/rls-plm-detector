"""
Signal processing functions for EMG preprocessing according to WASM 2019 guidelines.

This module implements the preprocessing pipeline for EMG signals:
1. Bandpass filtering (10-100 Hz)
2. Signal rectification
3. Moving average smoothing (300ms window)
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
from typing import Tuple, Optional, Dict, Any
import logging

from ..utils.data_structures import EMGSignal, SignalQuality, AnalysisConfig

logger = logging.getLogger(__name__)


def preprocess_emg(emg_signal: EMGSignal, config: Optional[AnalysisConfig] = None) -> EMGSignal:
    """
    Complete EMG preprocessing pipeline according to WASM 2019 guidelines.
    
    Args:
        emg_signal: Raw EMG signal
        config: Analysis configuration with preprocessing parameters
        
    Returns:
        Preprocessed EMG signal
    """
    config = config or AnalysisConfig()
    
    logger.info(f"Starting EMG preprocessing for channel: {emg_signal.channel_name}")
    
    # Step 1: Apply bandpass filter
    filtered_data = apply_bandpass_filter(
        emg_signal.data,
        emg_signal.sampling_frequency,
        config.emg_bandpass_low,
        config.emg_bandpass_high
    )
    
    # Step 2: Rectify the signal
    rectified_data = rectify_signal(filtered_data)
    
    # Step 3: Apply smoothing
    window_samples = int(config.smoothing_window_ms * emg_signal.sampling_frequency / 1000)
    smoothed_data = smooth_signal(rectified_data, window_samples)
    
    # Create preprocessed signal
    preprocessed_signal = EMGSignal(
        data=smoothed_data,
        sampling_frequency=emg_signal.sampling_frequency,
        channel_name=emg_signal.channel_name,
        units=emg_signal.units,
        start_time=emg_signal.start_time
    )
    
    # Assess signal quality
    quality = assess_signal_quality(preprocessed_signal)
    logger.info(f"Signal quality assessment: {quality}")
    
    logger.info(f"Completed EMG preprocessing for channel: {emg_signal.channel_name}")
    
    return preprocessed_signal


def apply_bandpass_filter(data: np.ndarray, fs: float, low_freq: float, high_freq: float, 
                         order: int = 4) -> np.ndarray:
    """
    Apply bandpass filter to EMG signal.
    
    Args:
        data: Input signal data
        fs: Sampling frequency
        low_freq: Lower cutoff frequency (Hz)
        high_freq: Upper cutoff frequency (Hz)
        order: Filter order
        
    Returns:
        Filtered signal data
    """
    # Normalize frequencies
    nyquist = fs / 2
    low_norm = low_freq / nyquist
    high_norm = high_freq / nyquist
    
    # Design Butterworth bandpass filter
    b, a = butter(order, [low_norm, high_norm], btype='band')
    
    # Apply filter with zero-phase filtering
    filtered_data = filtfilt(b, a, data)
    
    logger.debug(f"Applied bandpass filter: {low_freq}-{high_freq} Hz")
    
    return filtered_data


def rectify_signal(data: np.ndarray) -> np.ndarray:
    """
    Rectify the EMG signal (take absolute value).
    
    Args:
        data: Input signal data
        
    Returns:
        Rectified signal data
    """
    rectified_data = np.abs(data)
    
    logger.debug("Applied signal rectification")
    
    return rectified_data


def smooth_signal(data: np.ndarray, window_samples: int) -> np.ndarray:
    """
    Apply moving average smoothing to the signal.
    
    Args:
        data: Input signal data
        window_samples: Number of samples in smoothing window
        
    Returns:
        Smoothed signal data
    """
    if window_samples <= 1:
        return data
    
    # Use convolution for moving average
    window = np.ones(window_samples) / window_samples
    smoothed_data = np.convolve(data, window, mode='same')
    
    logger.debug(f"Applied smoothing with {window_samples} sample window")
    
    return smoothed_data


def assess_signal_quality(emg_signal: EMGSignal) -> SignalQuality:
    """
    Assess the quality of the preprocessed EMG signal.
    
    Args:
        emg_signal: Preprocessed EMG signal
        
    Returns:
        Signal quality metrics
    """
    data = emg_signal.data
    
    # Calculate signal-to-noise ratio
    signal_power = np.mean(data ** 2)
    noise_power = np.var(data)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
    
    # Calculate baseline stability
    baseline_std = np.std(data)
    baseline_stability = 1.0 / (1.0 + baseline_std)
    
    # Calculate artifact level (simplified approach)
    threshold = np.mean(data) + 3 * np.std(data)
    artifact_samples = np.sum(data > threshold)
    artifact_level = artifact_samples / len(data)
    
    # Calculate overall quality score
    overall_quality = (snr/100 + baseline_stability + (1-artifact_level)) / 3
    
    quality = SignalQuality(
        snr=snr,
        baseline_stability=baseline_stability,
        artifact_level=artifact_level,
        overall_quality=overall_quality
    )
    
    return quality


def remove_dc_offset(data: np.ndarray) -> np.ndarray:
    """
    Remove DC offset from the signal.
    
    Args:
        data: Input signal data
        
    Returns:
        Signal data with DC offset removed
    """
    dc_removed = data - np.mean(data)
    
    logger.debug("Removed DC offset")
    
    return dc_removed


def normalize_signal(data: np.ndarray, method: str = 'zscore') -> np.ndarray:
    """
    Normalize the signal using specified method.
    
    Args:
        data: Input signal data
        method: Normalization method ('zscore', 'minmax', 'rms')
        
    Returns:
        Normalized signal data
    """
    if method == 'zscore':
        normalized = (data - np.mean(data)) / np.std(data)
    elif method == 'minmax':
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    elif method == 'rms':
        rms = np.sqrt(np.mean(data ** 2))
        normalized = data / rms if rms > 0 else data
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    logger.debug(f"Applied {method} normalization")
    
    return normalized


def detect_artifacts(data: np.ndarray, threshold_factor: float = 3.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect artifacts in the EMG signal.
    
    Args:
        data: Input signal data
        threshold_factor: Factor for artifact detection threshold
        
    Returns:
        Tuple of (cleaned_data, artifact_info)
    """
    # Calculate threshold
    threshold = np.mean(data) + threshold_factor * np.std(data)
    
    # Find artifact samples
    artifact_mask = data > threshold
    
    # Clean artifacts by interpolation
    cleaned_data = data.copy()
    if np.any(artifact_mask):
        # Simple artifact cleaning: replace with median
        median_val = np.median(data[~artifact_mask])
        cleaned_data[artifact_mask] = median_val
    
    artifact_info = {
        'artifact_samples': np.sum(artifact_mask),
        'artifact_percentage': np.sum(artifact_mask) / len(data) * 100,
        'threshold': threshold,
        'cleaned': np.any(artifact_mask)
    }
    
    if artifact_info['cleaned']:
        logger.info(f"Cleaned {artifact_info['artifact_samples']} artifact samples "
                   f"({artifact_info['artifact_percentage']:.1f}%)")
    
    return cleaned_data, artifact_info


def validate_preprocessing_parameters(config: AnalysisConfig) -> bool:
    """
    Validate preprocessing parameters according to WASM 2019 guidelines.
    
    Args:
        config: Analysis configuration
        
    Returns:
        True if parameters are valid
    """
    errors = []
    
    # Check bandpass filter parameters
    if config.emg_bandpass_low >= config.emg_bandpass_high:
        errors.append("Low frequency must be less than high frequency")
    
    if config.emg_bandpass_low < 5 or config.emg_bandpass_high > 200:
        errors.append("Bandpass frequencies should be between 5-200 Hz")
    
    # Check smoothing window
    if config.smoothing_window_ms <= 0 or config.smoothing_window_ms > 1000:
        errors.append("Smoothing window should be between 0-1000 ms")
    
    if errors:
        for error in errors:
            logger.error(f"Parameter validation error: {error}")
        return False
    
    return True 