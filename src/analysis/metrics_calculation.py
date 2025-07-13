"""
Metrics calculation for WASM 2019 leg movement detection software.

This module implements STEP 10 of the WASM 2019 algorithm:
- PLMS index calculation (per hour)
- PLMW index calculation (wake movements per hour)
- Periodicity index calculation
- Mean duration calculation
- IMI distribution analysis
- Comprehensive diagnostic metrics
"""

import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from ..utils.data_structures import (
    CandidateLegMovement, PLMSequence, SleepStage, SleepStageSegment,
    AnalysisMetrics, DiagnosticResult, AnalysisConfig
)

logger = logging.getLogger(__name__)


def compute_analysis_metrics(clms: List[CandidateLegMovement],
                           plm_sequences: List[PLMSequence],
                           sleep_stages: List[SleepStageSegment],
                           config: Optional[AnalysisConfig] = None) -> AnalysisMetrics:
    """
    Compute comprehensive analysis metrics (STEP 10).
    
    Args:
        clms: List of candidate leg movements
        plm_sequences: List of PLM sequences
        sleep_stages: List of sleep stage segments
        config: Analysis configuration
        
    Returns:
        Comprehensive analysis metrics
    """
    config = config or AnalysisConfig()
    
    logger.info("Computing comprehensive analysis metrics")
    
    # Calculate sleep and wake durations
    stage_durations = calculate_stage_durations(sleep_stages)
    total_sleep_time = sum(duration for stage, duration in stage_durations.items() 
                          if stage != SleepStage.WAKE)
    total_wake_time = stage_durations[SleepStage.WAKE]
    
    # Calculate PLM indices
    plms_index = calculate_plms_index(plm_sequences, total_sleep_time)
    plmw_index = calculate_plmw_index(clms, sleep_stages, total_wake_time)
    
    # Calculate movement statistics
    total_clms = len(clms)
    total_plms = sum(len(seq.movements) for seq in plm_sequences)
    mean_clm_duration = np.mean([clm.duration for clm in clms]) if clms else 0.0
    mean_plm_duration = np.mean([clm.duration for clm in clms 
                                if any(clm in seq.movements for seq in plm_sequences)]) if clms else 0.0
    
    # Calculate periodicity
    periodicity_index = calculate_periodicity_index(plm_sequences, clms)
    mean_imi = np.mean([imi for seq in plm_sequences 
                       for i in range(1, len(seq.movements))
                       for imi in [seq.movements[i].start_time - seq.movements[i-1].start_time]]) if plm_sequences else 0.0
    imi_std = np.std([imi for seq in plm_sequences 
                     for i in range(1, len(seq.movements))
                     for imi in [seq.movements[i].start_time - seq.movements[i-1].start_time]]) if plm_sequences else 0.0
    
    # Calculate association indices
    respiratory_association_index = sum(1 for clm in clms if clm.associated_respiratory) / max(total_sleep_time / 3600, 0.001)
    arousal_association_index = sum(1 for clm in clms if clm.associated_arousal) / max(total_sleep_time / 3600, 0.001)
    
    # Calculate stage-specific distributions
    clms_by_stage = calculate_clms_by_stage(clms, sleep_stages)
    plms_by_stage = calculate_plms_by_stage(plm_sequences, sleep_stages)
    
    # Calculate IMI distribution
    imi_distribution = calculate_imi_distribution(plm_sequences)
    
    # Calculate quality scores
    signal_quality_score = 1.0  # Placeholder - should be calculated from signal quality
    baseline_stability_score = 1.0  # Placeholder - should be calculated from baseline analysis
    
    metrics = AnalysisMetrics(
        plms_index=plms_index,
        plmw_index=plmw_index,
        total_clms=total_clms,
        total_plms=total_plms,
        mean_clm_duration=mean_clm_duration,
        mean_plm_duration=mean_plm_duration,
        periodicity_index=periodicity_index,
        mean_imi=mean_imi,
        imi_std=imi_std,
        respiratory_association_index=respiratory_association_index,
        arousal_association_index=arousal_association_index,
        clms_by_stage=clms_by_stage,
        plms_by_stage=plms_by_stage,
        imi_distribution=imi_distribution,
        signal_quality_score=signal_quality_score,
        baseline_stability_score=baseline_stability_score
    )
    
    logger.info(f"Computed metrics: PLMS index={plms_index:.2f}, "
                f"PLMW index={plmw_index:.2f}, periodicity={periodicity_index:.2f}")
    
    return metrics


def calculate_plms_index(plm_sequences: List[PLMSequence], total_sleep_time: float) -> float:
    """
    Calculate PLMS index (PLMs per hour of sleep).
    
    Args:
        plm_sequences: List of PLM sequences
        total_sleep_time: Total sleep time in seconds
        
    Returns:
        PLMS index (per hour)
    """
    total_plms = sum(len(seq.movements) for seq in plm_sequences)
    sleep_hours = total_sleep_time / 3600
    
    plms_index = total_plms / max(sleep_hours, 0.001)
    
    return plms_index


def calculate_plmw_index(clms: List[CandidateLegMovement], 
                        sleep_stages: List[SleepStageSegment],
                        total_wake_time: float) -> float:
    """
    Calculate PLMW index (PLMs per hour of wake).
    
    Args:
        clms: List of candidate leg movements
        sleep_stages: List of sleep stage segments
        total_wake_time: Total wake time in seconds
        
    Returns:
        PLMW index (per hour)
    """
    # Count CLMs that occurred during wake
    wake_clms = 0
    sorted_stages = sorted(sleep_stages, key=lambda x: x.start_time)
    
    for clm in clms:
        for stage_segment in sorted_stages:
            if (stage_segment.stage == SleepStage.WAKE and
                clm.start_time >= stage_segment.start_time and
                clm.start_time < stage_segment.end_time):
                wake_clms += 1
                break
    
    wake_hours = total_wake_time / 3600
    plmw_index = wake_clms / max(wake_hours, 0.001)
    
    return plmw_index


def calculate_periodicity_index(plm_sequences: List[PLMSequence], 
                              clms: List[CandidateLegMovement]) -> float:
    """
    Calculate periodicity index (PLMs / CLMs).
    
    Args:
        plm_sequences: List of PLM sequences
        clms: List of candidate leg movements
        
    Returns:
        Periodicity index
    """
    total_plms = sum(len(seq.movements) for seq in plm_sequences)
    total_clms = len(clms)
    
    periodicity_index = total_plms / max(total_clms, 1)
    
    return periodicity_index


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


def calculate_clms_by_stage(clms: List[CandidateLegMovement], 
                           sleep_stages: List[SleepStageSegment]) -> Dict[SleepStage, int]:
    """
    Calculate CLM count by sleep stage.
    
    Args:
        clms: List of candidate leg movements
        sleep_stages: List of sleep stage segments
        
    Returns:
        Dictionary mapping sleep stages to CLM counts
    """
    clms_by_stage = {
        SleepStage.WAKE: 0,
        SleepStage.N1: 0,
        SleepStage.N2: 0,
        SleepStage.N3: 0,
        SleepStage.REM: 0,
        SleepStage.UNKNOWN: 0
    }
    
    sorted_stages = sorted(sleep_stages, key=lambda x: x.start_time)
    
    for clm in clms:
        assigned = False
        for stage_segment in sorted_stages:
            if (clm.start_time >= stage_segment.start_time and
                clm.start_time < stage_segment.end_time):
                clms_by_stage[stage_segment.stage] += 1
                assigned = True
                break
        
        if not assigned:
            clms_by_stage[SleepStage.UNKNOWN] += 1
    
    return clms_by_stage


def calculate_plms_by_stage(plm_sequences: List[PLMSequence], 
                           sleep_stages: List[SleepStageSegment]) -> Dict[SleepStage, int]:
    """
    Calculate PLM count by sleep stage.
    
    Args:
        plm_sequences: List of PLM sequences
        sleep_stages: List of sleep stage segments
        
    Returns:
        Dictionary mapping sleep stages to PLM counts
    """
    plms_by_stage = {
        SleepStage.WAKE: 0,
        SleepStage.N1: 0,
        SleepStage.N2: 0,
        SleepStage.N3: 0,
        SleepStage.REM: 0,
        SleepStage.UNKNOWN: 0
    }
    
    sorted_stages = sorted(sleep_stages, key=lambda x: x.start_time)
    
    for sequence in plm_sequences:
        for movement in sequence.movements:
            assigned = False
            for stage_segment in sorted_stages:
                if (movement.start_time >= stage_segment.start_time and
                    movement.start_time < stage_segment.end_time):
                    plms_by_stage[stage_segment.stage] += 1
                    assigned = True
                    break
            
            if not assigned:
                plms_by_stage[SleepStage.UNKNOWN] += 1
    
    return plms_by_stage


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


def generate_diagnostic_result(metrics: AnalysisMetrics,
                             clms: List[CandidateLegMovement],
                             plm_sequences: List[PLMSequence],
                             sleep_stages: List[SleepStageSegment],
                             patient_id: str = "unknown",
                             recording_date: str = "",
                             config: Optional[AnalysisConfig] = None) -> DiagnosticResult:
    """
    Generate final diagnostic result based on metrics.
    
    Args:
        metrics: Analysis metrics
        clms: List of candidate leg movements
        plm_sequences: List of PLM sequences
        sleep_stages: List of sleep stage segments
        patient_id: Patient identifier
        recording_date: Recording date
        config: Analysis configuration
        
    Returns:
        Diagnostic result
    """
    config = config or AnalysisConfig()
    
    logger.info("Generating diagnostic result")
    
    # Calculate total times
    stage_durations = calculate_stage_durations(sleep_stages)
    total_sleep_time = sum(duration for stage, duration in stage_durations.items() 
                          if stage != SleepStage.WAKE)
    total_wake_time = stage_durations[SleepStage.WAKE]
    
    # Determine PLMS diagnosis
    plms_diagnosis = determine_plms_diagnosis(metrics.plms_index, config)
    
    # Determine clinical significance
    clinical_significance = determine_clinical_significance(metrics, config)
    
    # Generate recommendations
    recommendations = generate_recommendations(metrics, plms_diagnosis, config)
    
    # Create diagnostic result
    diagnostic_result = DiagnosticResult(
        patient_id=patient_id,
        recording_date=recording_date,
        analysis_date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        metrics=metrics,
        all_clms=clms,
        all_plms=plm_sequences,
        total_sleep_time=total_sleep_time,
        total_wake_time=total_wake_time,
        sleep_stages=sleep_stages,
        plms_diagnosis=plms_diagnosis,
        clinical_significance=clinical_significance,
        recommendations=recommendations
    )
    
    logger.info(f"Generated diagnostic result: {plms_diagnosis}")
    
    return diagnostic_result


def determine_plms_diagnosis(plms_index: float, config: AnalysisConfig) -> str:
    """
    Determine PLMS diagnosis based on PLMS index.
    
    Args:
        plms_index: PLMS index value
        config: Analysis configuration
        
    Returns:
        Diagnosis string
    """
    if plms_index < config.plms_normal_threshold:
        return "normal"
    elif plms_index < config.plms_mild_threshold:
        return "mild"
    elif plms_index < config.plms_moderate_threshold:
        return "moderate"
    else:
        return "severe"


def determine_clinical_significance(metrics: AnalysisMetrics, config: AnalysisConfig) -> str:
    """
    Determine clinical significance of the findings.
    
    Args:
        metrics: Analysis metrics
        config: Analysis configuration
        
    Returns:
        Clinical significance description
    """
    if metrics.plms_index < config.plms_normal_threshold:
        return "No significant periodic leg movements detected"
    elif metrics.plms_index < config.plms_mild_threshold:
        return "Mild periodic leg movements - monitor for progression"
    elif metrics.plms_index < config.plms_moderate_threshold:
        return "Moderate periodic leg movements - consider treatment"
    else:
        return "Severe periodic leg movements - treatment recommended"


def generate_recommendations(metrics: AnalysisMetrics, diagnosis: str, 
                           config: AnalysisConfig) -> List[str]:
    """
    Generate clinical recommendations based on findings.
    
    Args:
        metrics: Analysis metrics
        diagnosis: PLMS diagnosis
        config: Analysis configuration
        
    Returns:
        List of recommendations
    """
    recommendations = []
    
    if diagnosis == "normal":
        recommendations.append("No specific treatment required for PLMS")
        recommendations.append("Monitor for any changes in symptoms")
    
    elif diagnosis == "mild":
        recommendations.append("Consider lifestyle modifications")
        recommendations.append("Monitor for symptom progression")
        recommendations.append("Evaluate for underlying conditions")
    
    elif diagnosis == "moderate":
        recommendations.append("Consider pharmacological treatment")
        recommendations.append("Address underlying conditions")
        recommendations.append("Monitor treatment response")
    
    elif diagnosis == "severe":
        recommendations.append("Immediate pharmacological treatment recommended")
        recommendations.append("Comprehensive evaluation for underlying conditions")
        recommendations.append("Regular follow-up monitoring")
        recommendations.append("Consider referral to sleep specialist")
    
    # Add general recommendations
    if metrics.respiratory_association_index > 5:
        recommendations.append("High respiratory association - evaluate for sleep apnea")
    
    if metrics.arousal_association_index > 10:
        recommendations.append("High arousal association - consider impact on sleep quality")
    
    return recommendations


def validate_metrics_parameters(config: AnalysisConfig) -> bool:
    """
    Validate metrics calculation parameters.
    
    Args:
        config: Analysis configuration
        
    Returns:
        True if parameters are valid
    """
    errors = []
    
    # Check diagnostic thresholds
    if config.plms_normal_threshold <= 0:
        errors.append("PLMS normal threshold must be positive")
    
    if config.plms_mild_threshold <= config.plms_normal_threshold:
        errors.append("PLMS mild threshold must be greater than normal threshold")
    
    if config.plms_moderate_threshold <= config.plms_mild_threshold:
        errors.append("PLMS moderate threshold must be greater than mild threshold")
    
    if errors:
        for error in errors:
            logger.error(f"Metrics parameter validation error: {error}")
        return False
    
    return True 