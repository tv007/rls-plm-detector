"""
Association analysis for leg movements according to WASM 2019 guidelines.

This module implements:
- STEP 7: Respiratory-associated CLMs (CLMr)
- STEP 8: Arousal-associated CLMs (PLMSa)
- STEP 9: Sleep stage segmentation
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging

from ..utils.data_structures import (
    CandidateLegMovement, RespiratoryEvent, ArousalEvent, 
    SleepStageSegment, SleepStage, AnalysisConfig
)

logger = logging.getLogger(__name__)


def associate_with_respiratory(clms: List[CandidateLegMovement], 
                             respiratory_events: List[RespiratoryEvent],
                             config: Optional[AnalysisConfig] = None) -> List[CandidateLegMovement]:
    """
    Associate CLMs with respiratory events (STEP 7).
    
    Args:
        clms: List of candidate leg movements
        respiratory_events: List of respiratory events
        config: Analysis configuration
        
    Returns:
        Updated CLMs with respiratory association markers
    """
    config = config or AnalysisConfig()
    
    logger.info("Associating CLMs with respiratory events")
    
    updated_clms = []
    
    for clm in clms:
        # Check if CLM is associated with any respiratory event
        associated = False
        
        for resp_event in respiratory_events:
            # Check temporal association: -2s to +10.25s from respiratory event end
            time_diff = clm.start_time - resp_event.end_time
            
            if (config.respiratory_pre_window <= time_diff <= config.respiratory_post_window):
                associated = True
                break
        
        # Create updated CLM with respiratory association
        updated_clm = CandidateLegMovement(
            start_time=clm.start_time,
            end_time=clm.end_time,
            duration=clm.duration,
            amplitude=clm.amplitude,
            baseline=clm.baseline,
            channel=clm.channel,
            side=clm.side,
            associated_respiratory=associated,
            associated_arousal=clm.associated_arousal
        )
        updated_clms.append(updated_clm)
    
    # Count associations
    respiratory_associations = sum(1 for clm in updated_clms if clm.associated_respiratory)
    logger.info(f"Found {respiratory_associations} respiratory-associated CLMs")
    
    return updated_clms


def associate_with_arousals(clms: List[CandidateLegMovement], 
                           arousal_events: List[ArousalEvent],
                           config: Optional[AnalysisConfig] = None) -> List[CandidateLegMovement]:
    """
    Associate CLMs with arousal events (STEP 8).
    
    Args:
        clms: List of candidate leg movements
        arousal_events: List of arousal events
        config: Analysis configuration
        
    Returns:
        Updated CLMs with arousal association markers
    """
    config = config or AnalysisConfig()
    
    logger.info("Associating CLMs with arousal events")
    
    updated_clms = []
    
    for clm in clms:
        # Check if CLM is associated with any arousal event
        associated = False
        
        for arousal in arousal_events:
            # Check for overlap or temporal proximity (≤0.5s)
            if (clm.start_time <= arousal.end_time and 
                clm.end_time >= arousal.start_time):  # Overlap
                associated = True
                break
            
            # Check temporal proximity
            time_diff = abs(clm.start_time - arousal.start_time)
            if time_diff <= config.arousal_proximity:
                associated = True
                break
        
        # Create updated CLM with arousal association
        updated_clm = CandidateLegMovement(
            start_time=clm.start_time,
            end_time=clm.end_time,
            duration=clm.duration,
            amplitude=clm.amplitude,
            baseline=clm.baseline,
            channel=clm.channel,
            side=clm.side,
            associated_respiratory=clm.associated_respiratory,
            associated_arousal=associated
        )
        updated_clms.append(updated_clm)
    
    # Count associations
    arousal_associations = sum(1 for clm in updated_clms if clm.associated_arousal)
    logger.info(f"Found {arousal_associations} arousal-associated CLMs")
    
    return updated_clms


def split_by_sleep_stage(clms: List[CandidateLegMovement], 
                        sleep_stages: List[SleepStageSegment]) -> Dict[SleepStage, List[CandidateLegMovement]]:
    """
    Split CLMs by sleep stage (STEP 9).
    
    Args:
        clms: List of candidate leg movements
        sleep_stages: List of sleep stage segments
        
    Returns:
        Dictionary mapping sleep stages to associated CLMs
    """
    logger.info("Splitting CLMs by sleep stage")
    
    # Initialize stage-specific CLM lists
    clms_by_stage = {
        SleepStage.WAKE: [],
        SleepStage.N1: [],
        SleepStage.N2: [],
        SleepStage.N3: [],
        SleepStage.REM: [],
        SleepStage.UNKNOWN: []
    }
    
    # Sort sleep stages by start time
    sorted_stages = sorted(sleep_stages, key=lambda x: x.start_time)
    
    # Assign each CLM to appropriate sleep stage
    for clm in clms:
        assigned = False
        
        for stage_segment in sorted_stages:
            # Check if CLM falls within this sleep stage segment
            if (clm.start_time >= stage_segment.start_time and 
                clm.start_time < stage_segment.end_time):
                
                clms_by_stage[stage_segment.stage].append(clm)
                assigned = True
                break
        
        # If not assigned to any stage, put in UNKNOWN
        if not assigned:
            clms_by_stage[SleepStage.UNKNOWN].append(clm)
    
    # Log distribution
    for stage, stage_clms in clms_by_stage.items():
        logger.info(f"Stage {stage.value}: {len(stage_clms)} CLMs")
    
    return clms_by_stage


def calculate_stage_durations(sleep_stages: List[SleepStageSegment]) -> Dict[SleepStage, float]:
    """
    Calculate total duration for each sleep stage.
    
    Args:
        sleep_stages: List of sleep stage segments
        
    Returns:
        Dictionary mapping sleep stages to total durations
    """
    stage_durations = {
        SleepStage.WAKE: 0.0,
        SleepStage.N1: 0.0,
        SleepStage.N2: 0.0,
        SleepStage.N3: 0.0,
        SleepStage.REM: 0.0,
        SleepStage.UNKNOWN: 0.0
    }
    
    for stage_segment in sleep_stages:
        stage_durations[stage_segment.stage] += stage_segment.duration
    
    return stage_durations


def calculate_association_indices(clms: List[CandidateLegMovement], 
                                total_sleep_time: float,
                                total_wake_time: float) -> Dict[str, float]:
    """
    Calculate association indices for respiratory and arousal events.
    
    Args:
        clms: List of candidate leg movements
        total_sleep_time: Total sleep time in seconds
        total_wake_time: Total wake time in seconds
        
    Returns:
        Dictionary of association indices
    """
    # Count associations
    respiratory_associations = sum(1 for clm in clms if clm.associated_respiratory)
    arousal_associations = sum(1 for clm in clms if clm.associated_arousal)
    
    # Calculate indices per hour
    sleep_hours = total_sleep_time / 3600
    wake_hours = total_wake_time / 3600
    
    respiratory_index = respiratory_associations / max(sleep_hours, 0.001)
    arousal_index = arousal_associations / max(sleep_hours, 0.001)
    
    return {
        'respiratory_association_index': respiratory_index,
        'arousal_association_index': arousal_index,
        'respiratory_association_count': respiratory_associations,
        'arousal_association_count': arousal_associations
    }


def validate_association_parameters(config: AnalysisConfig) -> bool:
    """
    Validate association analysis parameters.
    
    Args:
        config: Analysis configuration
        
    Returns:
        True if parameters are valid
    """
    errors = []
    
    # Check respiratory association windows
    if config.respiratory_pre_window < 0:
        errors.append("Respiratory pre-window must be non-negative")
    
    if config.respiratory_post_window <= 0:
        errors.append("Respiratory post-window must be positive")
    
    # Check arousal proximity
    if config.arousal_proximity <= 0:
        errors.append("Arousal proximity must be positive")
    
    if errors:
        for error in errors:
            logger.error(f"Association parameter validation error: {error}")
        return False
    
    return True


def assess_association_quality(clms: List[CandidateLegMovement],
                             respiratory_events: List[RespiratoryEvent],
                             arousal_events: List[ArousalEvent]) -> Dict[str, Any]:
    """
    Assess the quality of association analysis.
    
    Args:
        clms: List of candidate leg movements
        respiratory_events: List of respiratory events
        arousal_events: List of arousal events
        
    Returns:
        Quality assessment dictionary
    """
    total_clms = len(clms)
    respiratory_associations = sum(1 for clm in clms if clm.associated_respiratory)
    arousal_associations = sum(1 for clm in clms if clm.associated_arousal)
    
    quality_metrics = {
        'total_clms': total_clms,
        'respiratory_associations': respiratory_associations,
        'arousal_associations': arousal_associations,
        'respiratory_association_rate': respiratory_associations / max(total_clms, 1),
        'arousal_association_rate': arousal_associations / max(total_clms, 1),
        'total_respiratory_events': len(respiratory_events),
        'total_arousal_events': len(arousal_events)
    }
    
    return quality_metrics


def create_association_summary(clms: List[CandidateLegMovement]) -> Dict[str, Any]:
    """
    Create a summary of association analysis results.
    
    Args:
        clms: List of candidate leg movements with associations
        
    Returns:
        Summary dictionary
    """
    summary = {
        'total_clms': len(clms),
        'respiratory_associated': 0,
        'arousal_associated': 0,
        'both_associated': 0,
        'neither_associated': 0
    }
    
    for clm in clms:
        if clm.associated_respiratory and clm.associated_arousal:
            summary['both_associated'] += 1
        elif clm.associated_respiratory:
            summary['respiratory_associated'] += 1
        elif clm.associated_arousal:
            summary['arousal_associated'] += 1
        else:
            summary['neither_associated'] += 1
    
    return summary 