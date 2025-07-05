"""
Data structures for WASM 2019 leg movement detection software.

This module defines all the core data structures used throughout the analysis pipeline.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from pydantic import BaseModel, Field, validator
import pandas as pd


class SleepStage(Enum):
    """Sleep stage enumeration according to AASM guidelines."""
    WAKE = "wake"
    N1 = "n1"
    N2 = "n2"
    N3 = "n3"
    REM = "rem"
    UNKNOWN = "unknown"


class EventType(Enum):
    """Event type enumeration for different types of leg movements."""
    LM = "leg_movement"
    CLM = "candidate_leg_movement"
    PLM = "periodic_leg_movement"
    CLMR = "respiratory_associated_clm"
    PLMSA = "arousal_associated_plm"


@dataclass
class EMGSignal:
    """EMG signal data structure."""
    data: np.ndarray
    sampling_frequency: float
    channel_name: str
    units: str = "µV"
    start_time: float = 0.0
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = len(self.data) / self.sampling_frequency
    
    @property
    def time_axis(self) -> np.ndarray:
        """Get time axis for the signal."""
        return np.arange(len(self.data)) / self.sampling_frequency + self.start_time


@dataclass
class EEGSignal:
    """EEG signal data structure."""
    data: np.ndarray
    sampling_frequency: float
    channel_name: str
    units: str = "µV"
    start_time: float = 0.0
    duration: Optional[float] = None
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = len(self.data) / self.sampling_frequency


@dataclass
class RespiratoryEvent:
    """Respiratory event data structure."""
    start_time: float
    end_time: float
    event_type: str  # "apnea", "hypopnea", etc.
    duration: float
    severity: Optional[str] = None
    associated_arousal: bool = False
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end_time - self.start_time


@dataclass
class ArousalEvent:
    """Arousal event data structure."""
    start_time: float
    end_time: float
    duration: float
    arousal_type: str = "spontaneous"
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end_time - self.start_time


@dataclass
class LegMovement:
    """Individual leg movement event."""
    start_time: float
    end_time: float
    duration: float
    amplitude: float
    baseline: float
    channel: str
    event_type: EventType = EventType.LM
    associated_respiratory: bool = False
    associated_arousal: bool = False
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end_time - self.start_time
    
    @property
    def amplitude_above_baseline(self) -> float:
        """Get amplitude above baseline."""
        return self.amplitude - self.baseline


@dataclass
class CandidateLegMovement:
    """Candidate leg movement after filtering."""
    start_time: float
    end_time: float
    duration: float
    amplitude: float
    baseline: float
    channel: str
    side: str  # "left", "right", "bilateral"
    associated_respiratory: bool = False
    associated_arousal: bool = False
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end_time - self.start_time


@dataclass
class PLMSequence:
    """Periodic leg movement sequence."""
    movements: List[CandidateLegMovement]
    start_time: float
    end_time: float
    duration: float
    movement_count: int
    mean_imi: float
    imi_std: float
    periodicity_index: float
    
    def __post_init__(self):
        if not self.movements:
            raise ValueError("PLM sequence must contain at least one movement")
        
        if self.start_time is None:
            self.start_time = min(m.start_time for m in self.movements)
        
        if self.end_time is None:
            self.end_time = max(m.end_time for m in self.movements)
        
        if self.duration is None:
            self.duration = self.end_time - self.start_time
        
        if self.movement_count is None:
            self.movement_count = len(self.movements)
        
        # Calculate IMI statistics
        if len(self.movements) > 1:
            imis = []
            for i in range(1, len(self.movements)):
                imi = self.movements[i].start_time - self.movements[i-1].start_time
                imis.append(imi)
            
            self.mean_imi = np.mean(imis)
            self.imi_std = np.std(imis)
        else:
            self.mean_imi = 0.0
            self.imi_std = 0.0


@dataclass
class SleepStageSegment:
    """Sleep stage segment with associated events."""
    stage: SleepStage
    start_time: float
    end_time: float
    duration: float
    clms: List[CandidateLegMovement] = field(default_factory=list)
    plms: List[PLMSequence] = field(default_factory=list)
    
    def __post_init__(self):
        if self.duration is None:
            self.duration = self.end_time - self.start_time


@dataclass
class AnalysisMetrics:
    """Comprehensive analysis metrics."""
    # PLM indices
    plms_index: float  # PLMs per hour of sleep
    plmw_index: float  # PLMs per hour of wake
    
    # Movement statistics
    total_clms: int
    total_plms: int
    mean_clm_duration: float
    mean_plm_duration: float
    
    # Periodicity
    periodicity_index: float  # PLMs / CLMs
    mean_imi: float
    imi_std: float
    
    # Association indices
    respiratory_association_index: float
    arousal_association_index: float
    
    # Sleep stage distribution
    clms_by_stage: Dict[SleepStage, int] = field(default_factory=dict)
    plms_by_stage: Dict[SleepStage, int] = field(default_factory=dict)
    
    # IMI distribution
    imi_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Quality metrics
    signal_quality_score: float = 1.0
    baseline_stability_score: float = 1.0


@dataclass
class DiagnosticResult:
    """Final diagnostic result."""
    patient_id: str
    recording_date: str
    analysis_date: str
    
    # Metrics
    metrics: AnalysisMetrics
    
    # Events
    all_clms: List[CandidateLegMovement]
    all_plms: List[PLMSequence]
    
    # Sleep data
    total_sleep_time: float
    total_wake_time: float
    sleep_stages: List[SleepStageSegment]
    
    # Diagnostic classification
    plms_diagnosis: str  # "normal", "mild", "moderate", "severe"
    clinical_significance: str
    recommendations: List[str] = field(default_factory=list)
    
    # Quality indicators
    analysis_quality: str = "good"
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SignalQuality(BaseModel):
    """Signal quality assessment model."""
    snr: float = Field(..., description="Signal-to-noise ratio")
    baseline_stability: float = Field(..., description="Baseline stability score (0-1)")
    artifact_level: float = Field(..., description="Artifact level (0-1)")
    overall_quality: float = Field(..., description="Overall quality score (0-1)")
    
    @validator('overall_quality')
    def validate_quality(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Quality score must be between 0 and 1')
        return v


class AnalysisConfig(BaseModel):
    """Configuration for analysis parameters."""
    # Preprocessing
    emg_bandpass_low: float = 10.0
    emg_bandpass_high: float = 100.0
    smoothing_window_ms: float = 300.0
    
    # Baseline estimation
    initial_baseline_duration: float = 30.0  # seconds
    baseline_window_duration: float = 15.0   # seconds
    baseline_high_warning: float = 5.0       # µV
    baseline_high_caution: float = 16.0      # µV
    
    # Leg movement detection
    amplitude_threshold: float = 8.0         # µV above baseline
    minimum_duration: float = 0.5            # seconds
    maximum_duration: float = 10.0           # seconds
    median_threshold: float = 2.0            # µV above baseline
    
    # PLM detection
    min_imi: float = 10.0                    # seconds
    max_imi: float = 90.0                    # seconds
    min_plm_sequence_length: int = 4
    
    # Association windows
    respiratory_pre_window: float = 2.0      # seconds
    respiratory_post_window: float = 10.25   # seconds
    arousal_proximity: float = 0.5           # seconds
    
    # Bilateral combination
    bilateral_tolerance: float = 0.5         # seconds
    max_combined_duration: float = 15.0      # seconds
    
    # Diagnostic thresholds
    plms_normal_threshold: float = 5.0       # per hour
    plms_mild_threshold: float = 15.0        # per hour
    plms_moderate_threshold: float = 25.0    # per hour
    
    @validator('emg_bandpass_low', 'emg_bandpass_high')
    def validate_bandpass(cls, v, values):
        if 'emg_bandpass_low' in values and v <= values['emg_bandpass_low']:
            raise ValueError('High frequency must be greater than low frequency')
        return v 