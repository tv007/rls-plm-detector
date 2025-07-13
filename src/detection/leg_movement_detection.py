"""
Leg movement detection algorithms according to WASM 2019 guidelines.

This module implements:
- STEP 3: Leg movement detection with amplitude and duration criteria
- STEP 4: Candidate leg movement filtering
- STEP 5: Bilateral CLM combination
- STEP 6: Periodic leg movement sequence detection
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import logging

from ..utils.data_structures import (
    EMGSignal, LegMovement, CandidateLegMovement, PLMSequence, 
    AnalysisConfig, EventType
)

logger = logging.getLogger(__name__)


def detect_leg_movements(emg_signal: EMGSignal, baseline: np.ndarray, 
                        config: Optional[AnalysisConfig] = None) -> List[LegMovement]:
    """
    Detect leg movements according to WASM 2019 guidelines (STEP 3).
    
    Args:
        emg_signal: Preprocessed EMG signal
        baseline: Dynamic baseline array
        config: Analysis configuration
        
    Returns:
        List of detected leg movements
    """
    config = config or AnalysisConfig()
    
    logger.info(f"Detecting leg movements for channel: {emg_signal.channel_name}")
    
    data = emg_signal.data
    fs = emg_signal.sampling_frequency
    leg_movements = []
    
    # Step 1: Find amplitude threshold crossings
    threshold = baseline + config.amplitude_threshold
    above_threshold = data >= threshold
    
    # Step 2: Find movement boundaries
    i = 0
    while i < len(data):
        if above_threshold[i]:
            # Start of potential movement
            start_idx = i
            
            # Find end of movement (below baseline + 2µV)
            end_threshold = baseline[i] + config.median_threshold
            while i < len(data) and data[i] >= end_threshold:
                i += 1
            
            end_idx = i
            
            # Calculate movement properties
            duration = (end_idx - start_idx) / fs
            amplitude = np.max(data[start_idx:end_idx])
            median_amplitude = np.median(data[start_idx:end_idx])
            
            # Validate movement criteria
            if (duration >= config.minimum_duration and 
                median_amplitude >= baseline[start_idx] + config.median_threshold):
                
                movement = LegMovement(
                    start_time=start_idx / fs,
                    end_time=end_idx / fs,
                    duration=duration,
                    amplitude=amplitude,
                    baseline=baseline[start_idx],
                    channel=emg_signal.channel_name
                )
                leg_movements.append(movement)
                
                logger.debug(f"Detected LM: {duration:.2f}s, amplitude: {amplitude:.2f}µV")
        else:
            i += 1
    
    logger.info(f"Detected {len(leg_movements)} leg movements")
    return leg_movements


def filter_candidate_leg_movements(leg_movements: List[LegMovement], 
                                 config: Optional[AnalysisConfig] = None) -> List[CandidateLegMovement]:
    """
    Filter leg movements to create candidate leg movements (STEP 4).
    
    Args:
        leg_movements: List of detected leg movements
        config: Analysis configuration
        
    Returns:
        List of candidate leg movements
    """
    config = config or AnalysisConfig()
    
    logger.info("Filtering candidate leg movements")
    
    candidate_movements = []
    
    for movement in leg_movements:
        # Check duration criteria: 0.5s ≤ duration ≤ 10s
        if config.minimum_duration <= movement.duration <= config.maximum_duration:
            
            # Determine side based on channel name
            side = determine_movement_side(movement.channel)
            
            candidate = CandidateLegMovement(
                start_time=movement.start_time,
                end_time=movement.end_time,
                duration=movement.duration,
                amplitude=movement.amplitude,
                baseline=movement.baseline,
                channel=movement.channel,
                side=side
            )
            candidate_movements.append(candidate)
    
    logger.info(f"Filtered to {len(candidate_movements)} candidate leg movements")
    return candidate_movements


def determine_movement_side(channel_name: str) -> str:
    """
    Determine the side (left/right) of the movement based on channel name.
    
    Args:
        channel_name: Name of the EMG channel
        
    Returns:
        Side identifier ('left', 'right', or 'unknown')
    """
    channel_lower = channel_name.lower()
    
    if any(keyword in channel_lower for keyword in ['left', 'l_', '_l']):
        return 'left'
    elif any(keyword in channel_lower for keyword in ['right', 'r_', '_r']):
        return 'right'
    else:
        return 'unknown'


def combine_bilateral_clms(clm_left: List[CandidateLegMovement], 
                          clm_right: List[CandidateLegMovement],
                          config: Optional[AnalysisConfig] = None) -> List[CandidateLegMovement]:
    """
    Combine bilateral candidate leg movements (STEP 5).
    
    Args:
        clm_left: Left side candidate leg movements
        clm_right: Right side candidate leg movements
        config: Analysis configuration
        
    Returns:
        List of combined bilateral CLMs
    """
    config = config or AnalysisConfig()
    
    logger.info("Combining bilateral candidate leg movements")
    
    bilateral_clms = []
    
    # Sort movements by start time
    clm_left_sorted = sorted(clm_left, key=lambda x: x.start_time)
    clm_right_sorted = sorted(clm_right, key=lambda x: x.start_time)
    
    # Find temporally aligned movements
    for clm_l in clm_left_sorted:
        for clm_r in clm_right_sorted:
            # Check temporal alignment (≤0.5s difference)
            time_diff = abs(clm_l.start_time - clm_r.start_time)
            
            if time_diff <= config.bilateral_tolerance:
                # Calculate combined duration
                combined_start = min(clm_l.start_time, clm_r.start_time)
                combined_end = max(clm_l.end_time, clm_r.end_time)
                combined_duration = combined_end - combined_start
                
                # Check combined duration criteria (≤15s)
                if (combined_duration <= config.max_combined_duration and
                    clm_l.duration <= config.maximum_duration and
                    clm_r.duration <= config.maximum_duration):
                    
                    # Create bilateral CLM
                    bilateral_clm = CandidateLegMovement(
                        start_time=combined_start,
                        end_time=combined_end,
                        duration=combined_duration,
                        amplitude=max(clm_l.amplitude, clm_r.amplitude),
                        baseline=max(clm_l.baseline, clm_r.baseline),
                        channel=f"{clm_l.channel}_{clm_r.channel}",
                        side="bilateral"
                    )
                    bilateral_clms.append(bilateral_clm)
                    
                    logger.debug(f"Combined bilateral CLM: {combined_duration:.2f}s")
    
    logger.info(f"Created {len(bilateral_clms)} bilateral CLMs")
    return bilateral_clms


def detect_plm_sequences(clms: List[CandidateLegMovement], 
                        config: Optional[AnalysisConfig] = None) -> List[PLMSequence]:
    """
    Detect periodic leg movement sequences (STEP 6).
    
    Args:
        clms: List of candidate leg movements
        config: Analysis configuration
        
    Returns:
        List of PLM sequences
    """
    config = config or AnalysisConfig()
    
    logger.info("Detecting periodic leg movement sequences")
    
    # Sort CLMs by start time
    sorted_clms = sorted(clms, key=lambda x: x.start_time)
    plm_sequences = []
    
    i = 0
    while i < len(sorted_clms) - config.min_plm_sequence_length + 1:
        # Start potential sequence
        sequence = [sorted_clms[i]]
        j = i + 1
        
        # Extend sequence while IMI criteria are met
        while j < len(sorted_clms):
            imi = sorted_clms[j].start_time - sorted_clms[j-1].start_time
            
            # Check IMI criteria: 10s ≤ IMI ≤ 90s
            if config.min_imi <= imi <= config.max_imi:
                sequence.append(sorted_clms[j])
                j += 1
            else:
                break
        
        # Check if sequence meets minimum length requirement
        if len(sequence) >= config.min_plm_sequence_length:
            # Create PLM sequence
            plm_sequence = create_plm_sequence(sequence)
            plm_sequences.append(plm_sequence)
            
            logger.debug(f"Detected PLM sequence: {len(sequence)} movements, "
                        f"duration: {plm_sequence.duration:.2f}s")
        
        i = j
    
    logger.info(f"Detected {len(plm_sequences)} PLM sequences")
    return plm_sequences


def create_plm_sequence(movements: List[CandidateLegMovement]) -> PLMSequence:
    """
    Create a PLM sequence from a list of movements.
    
    Args:
        movements: List of movements forming the sequence
        
    Returns:
        PLM sequence object
    """
    if not movements:
        raise ValueError("Cannot create PLM sequence from empty movement list")
    
    # Calculate sequence properties
    start_time = min(m.start_time for m in movements)
    end_time = max(m.end_time for m in movements)
    duration = end_time - start_time
    movement_count = len(movements)
    
    # Calculate IMI statistics
    imis = []
    for i in range(1, len(movements)):
        imi = movements[i].start_time - movements[i-1].start_time
        imis.append(imi)
    
    mean_imi = np.mean(imis) if imis else 0.0
    imi_std = np.std(imis) if imis else 0.0
    
    # Calculate periodicity index
    periodicity_index = movement_count / max(duration / 3600, 0.001)  # per hour
    
    plm_sequence = PLMSequence(
        movements=movements,
        start_time=start_time,
        end_time=end_time,
        duration=duration,
        movement_count=movement_count,
        mean_imi=mean_imi,
        imi_std=imi_std,
        periodicity_index=periodicity_index
    )
    
    return plm_sequence


def calculate_imi_distribution(plm_sequences: List[PLMSequence]) -> Dict[str, int]:
    """
    Calculate the distribution of inter-movement intervals.
    
    Args:
        plm_sequences: List of PLM sequences
        
    Returns:
        Dictionary of IMI distribution
    """
    all_imis = []
    
    for sequence in plm_sequences:
        for i in range(1, len(sequence.movements)):
            imi = sequence.movements[i].start_time - sequence.movements[i-1].start_time
            all_imis.append(imi)
    
    if not all_imis:
        return {}
    
    # Create histogram bins
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    hist, _ = np.histogram(all_imis, bins=bins)
    
    distribution = {}
    for i, count in enumerate(hist):
        bin_label = f"{bins[i]}-{bins[i+1] if i+1 < len(bins) else '90+'}s"
        distribution[bin_label] = int(count)
    
    return distribution


def validate_detection_parameters(config: AnalysisConfig) -> bool:
    """
    Validate detection parameters according to WASM 2019 guidelines.
    
    Args:
        config: Analysis configuration
        
    Returns:
        True if parameters are valid
    """
    errors = []
    
    # Check amplitude threshold
    if config.amplitude_threshold <= 0:
        errors.append("Amplitude threshold must be positive")
    
    # Check duration thresholds
    if config.minimum_duration <= 0 or config.maximum_duration <= 0:
        errors.append("Duration thresholds must be positive")
    
    if config.minimum_duration >= config.maximum_duration:
        errors.append("Minimum duration must be less than maximum duration")
    
    # Check median threshold
    if config.median_threshold <= 0:
        errors.append("Median threshold must be positive")
    
    # Check IMI thresholds
    if config.min_imi <= 0 or config.max_imi <= 0:
        errors.append("IMI thresholds must be positive")
    
    if config.min_imi >= config.max_imi:
        errors.append("Minimum IMI must be less than maximum IMI")
    
    # Check PLM sequence parameters
    if config.min_plm_sequence_length < 2:
        errors.append("Minimum PLM sequence length must be at least 2")
    
    # Check bilateral parameters
    if config.bilateral_tolerance <= 0:
        errors.append("Bilateral tolerance must be positive")
    
    if config.max_combined_duration <= 0:
        errors.append("Maximum combined duration must be positive")
    
    if errors:
        for error in errors:
            logger.error(f"Detection parameter validation error: {error}")
        return False
    
    return True


def assess_detection_quality(leg_movements: List[LegMovement], 
                           clms: List[CandidateLegMovement],
                           plm_sequences: List[PLMSequence]) -> Dict[str, Any]:
    """
    Assess the quality of the detection results.
    
    Args:
        leg_movements: Detected leg movements
        clms: Candidate leg movements
        plm_sequences: PLM sequences
        
    Returns:
        Quality assessment dictionary
    """
    quality_metrics = {
        'total_movements': len(leg_movements),
        'total_clms': len(clms),
        'total_plm_sequences': len(plm_sequences),
        'detection_rate': len(clms) / max(len(leg_movements), 1),
        'plm_rate': len(plm_sequences) / max(len(clms), 1),
        'average_clm_duration': np.mean([clm.duration for clm in clms]) if clms else 0,
        'average_plm_sequence_length': np.mean([len(seq.movements) for seq in plm_sequences]) if plm_sequences else 0
    }
    
    return quality_metrics 