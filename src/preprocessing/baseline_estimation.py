"""
Baseline estimation for EMG signals according to WASM 2019 guidelines.

This module implements STEP 2 of the WASM 2019 algorithm:
- Initial baseline calculation (first 30 seconds)
- Dynamic baseline estimation with 15s sliding windows
- Baseline validation and warning checks
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from ..utils.data_structures import EMGSignal, AnalysisConfig

logger = logging.getLogger(__name__)


def compute_baseline(emg_signal: EMGSignal, config: Optional[AnalysisConfig] = None) -> np.ndarray:
    """
    Compute dynamic baseline for EMG signal according to WASM 2019 guidelines.
    
    Args:
        emg_signal: Preprocessed EMG signal
        config: Analysis configuration with baseline parameters
        
    Returns:
        Dynamic baseline array with same length as signal
    """
    config = config or AnalysisConfig()
    
    logger.info(f"Computing baseline for channel: {emg_signal.channel_name}")
    
    # Step 1: Calculate initial baseline from first 30 seconds
    initial_baseline = compute_initial_baseline(emg_signal, config)
    
    # Step 2: Compute dynamic baseline with sliding windows
    dynamic_baseline = compute_dynamic_baseline(emg_signal, initial_baseline, config)
    
    # Step 3: Validate baseline
    validate_baseline(dynamic_baseline, config)
    
    logger.info(f"Completed baseline computation for channel: {emg_signal.channel_name}")
    
    return dynamic_baseline


def compute_initial_baseline(emg_signal: EMGSignal, config: AnalysisConfig) -> float:
    """
    Compute initial baseline from the first 30 seconds of the signal.
    
    Args:
        emg_signal: EMG signal
        config: Analysis configuration
        
    Returns:
        Initial baseline value
    """
    # Calculate samples for initial baseline period
    initial_samples = int(config.initial_baseline_duration * emg_signal.sampling_frequency)
    
    # Ensure we don't exceed signal length
    initial_samples = min(initial_samples, len(emg_signal.data))
    
    # Calculate mean of first 30 seconds
    initial_baseline = np.mean(emg_signal.data[:initial_samples])
    
    logger.info(f"Initial baseline: {initial_baseline:.2f} µV")
    
    # Check for high baseline warning
    if abs(initial_baseline) > config.baseline_high_warning:
        logger.warning(f"High initial baseline detected: {initial_baseline:.2f} µV")
    
    return initial_baseline


def compute_dynamic_baseline(emg_signal: EMGSignal, initial_baseline: float, 
                           config: AnalysisConfig) -> np.ndarray:
    """
    Compute dynamic baseline using sliding windows.
    
    Args:
        emg_signal: EMG signal
        initial_baseline: Initial baseline value
        config: Analysis configuration
        
    Returns:
        Dynamic baseline array
    """
    data = emg_signal.data
    fs = emg_signal.sampling_frequency
    
    # Calculate window size in samples
    window_samples = int(config.baseline_window_duration * fs)
    
    # Initialize baseline array
    baseline = np.full_like(data, initial_baseline)
    
    # Apply sliding window baseline estimation
    for i in range(0, len(data) - window_samples + 1, window_samples // 2):  # 50% overlap
        window_data = data[i:i + window_samples]
        
        # Check if window has low variability (stable baseline)
        window_std = np.std(window_data)
        
        if window_std < 3.0:  # µV threshold for stability
            # Update baseline for this window
            window_baseline = np.mean(window_data)
            end_idx = min(i + window_samples, len(baseline))
            baseline[i:end_idx] = window_baseline
    
    # Smooth the baseline to avoid abrupt changes
    baseline = smooth_baseline(baseline, fs)
    
    logger.info(f"Dynamic baseline computed with {window_samples} sample windows")
    
    return baseline


def smooth_baseline(baseline: np.ndarray, fs: float, smoothing_window_ms: float = 5000) -> np.ndarray:
    """
    Smooth the baseline to avoid abrupt changes.
    
    Args:
        baseline: Baseline array
        fs: Sampling frequency
        smoothing_window_ms: Smoothing window in milliseconds
        
    Returns:
        Smoothed baseline array
    """
    window_samples = int(smoothing_window_ms * fs / 1000)
    
    if window_samples <= 1:
        return baseline
    
    # Apply moving average smoothing
    window = np.ones(window_samples) / window_samples
    smoothed_baseline = np.convolve(baseline, window, mode='same')
    
    return smoothed_baseline


def validate_baseline(baseline: np.ndarray, config: AnalysisConfig) -> Dict[str, Any]:
    """
    Validate baseline according to WASM 2019 guidelines.
    
    Args:
        baseline: Baseline array
        config: Analysis configuration
        
    Returns:
        Validation results dictionary
    """
    validation_results = {
        'is_valid': True,
        'warnings': [],
        'errors': []
    }
    
    # Check for high baseline values
    high_baseline_mask = np.abs(baseline) > config.baseline_high_caution
    high_baseline_percentage = np.sum(high_baseline_mask) / len(baseline) * 100
    
    if high_baseline_percentage > 10:  # More than 10% of signal
        validation_results['warnings'].append(
            f"High baseline detected in {high_baseline_percentage:.1f}% of signal"
        )
        logger.warning(f"High baseline detected in {high_baseline_percentage:.1f}% of signal")
    
    # Check for baseline stability
    baseline_std = np.std(baseline)
    if baseline_std > 5.0:  # µV
        validation_results['warnings'].append(
            f"Unstable baseline detected (std: {baseline_std:.2f} µV)"
        )
        logger.warning(f"Unstable baseline detected (std: {baseline_std:.2f} µV)")
    
    # Check for extreme values
    if np.any(np.abs(baseline) > 50.0):  # µV
        validation_results['errors'].append("Extreme baseline values detected (>50 µV)")
        validation_results['is_valid'] = False
        logger.error("Extreme baseline values detected (>50 µV)")
    
    return validation_results


def estimate_baseline_quality(baseline: np.ndarray, emg_signal: EMGSignal) -> float:
    """
    Estimate the quality of the baseline estimation.
    
    Args:
        baseline: Baseline array
        emg_signal: Original EMG signal
        
    Returns:
        Quality score (0-1, higher is better)
    """
    # Calculate baseline stability
    baseline_stability = 1.0 / (1.0 + np.std(baseline))
    
    # Calculate baseline consistency with signal
    signal_mean = np.mean(emg_signal.data)
    baseline_mean = np.mean(baseline)
    consistency = 1.0 / (1.0 + abs(baseline_mean - signal_mean))
    
    # Calculate overall quality
    quality_score = (baseline_stability + consistency) / 2
    
    return quality_score


def detect_baseline_artifacts(baseline: np.ndarray, threshold_factor: float = 2.0) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Detect and correct artifacts in the baseline.
    
    Args:
        baseline: Baseline array
        threshold_factor: Factor for artifact detection
        
    Returns:
        Tuple of (corrected_baseline, artifact_info)
    """
    # Calculate threshold for artifact detection
    threshold = np.mean(baseline) + threshold_factor * np.std(baseline)
    
    # Find artifacts
    artifact_mask = np.abs(baseline) > threshold
    
    # Correct artifacts by interpolation
    corrected_baseline = baseline.copy()
    if np.any(artifact_mask):
        # Use median filtering for artifact correction
        from scipy.signal import medfilt
        corrected_baseline = medfilt(baseline, kernel_size=5)
    
    artifact_info = {
        'artifact_samples': np.sum(artifact_mask),
        'artifact_percentage': np.sum(artifact_mask) / len(baseline) * 100,
        'corrected': np.any(artifact_mask)
    }
    
    if artifact_info['corrected']:
        logger.info(f"Corrected {artifact_info['artifact_samples']} baseline artifacts "
                   f"({artifact_info['artifact_percentage']:.1f}%)")
    
    return corrected_baseline, artifact_info


def compute_adaptive_baseline(emg_signal: EMGSignal, config: AnalysisConfig) -> np.ndarray:
    """
    Compute adaptive baseline using more sophisticated methods.
    
    Args:
        emg_signal: EMG signal
        config: Analysis configuration
        
    Returns:
        Adaptive baseline array
    """
    data = emg_signal.data
    fs = emg_signal.sampling_frequency
    
    # Use percentile-based baseline estimation
    window_samples = int(config.baseline_window_duration * fs)
    baseline = np.zeros_like(data)
    
    for i in range(0, len(data), window_samples // 2):
        end_idx = min(i + window_samples, len(data))
        window_data = data[i:end_idx]
        
        # Use 10th percentile as baseline (more robust than mean)
        window_baseline = np.percentile(window_data, 10)
        baseline[i:end_idx] = window_baseline
    
    # Smooth the baseline
    baseline = smooth_baseline(baseline, fs)
    
    logger.info("Computed adaptive baseline using percentile method")
    
    return baseline


def validate_baseline_parameters(config: AnalysisConfig) -> bool:
    """
    Validate baseline estimation parameters.
    
    Args:
        config: Analysis configuration
        
    Returns:
        True if parameters are valid
    """
    errors = []
    
    # Check initial baseline duration
    if config.initial_baseline_duration <= 0 or config.initial_baseline_duration > 300:
        errors.append("Initial baseline duration should be between 0-300 seconds")
    
    # Check baseline window duration
    if config.baseline_window_duration <= 0 or config.baseline_window_duration > 60:
        errors.append("Baseline window duration should be between 0-60 seconds")
    
    # Check baseline thresholds
    if config.baseline_high_warning <= 0 or config.baseline_high_caution <= 0:
        errors.append("Baseline thresholds must be positive")
    
    if config.baseline_high_warning >= config.baseline_high_caution:
        errors.append("Warning threshold should be less than caution threshold")
    
    if errors:
        for error in errors:
            logger.error(f"Baseline parameter validation error: {error}")
        return False
    
    return True 