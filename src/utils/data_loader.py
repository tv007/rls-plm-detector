"""
Data loading utilities for WASM 2019 leg movement detection software.

This module provides functions to load and parse various sleep study data formats,
primarily EDF files containing EMG, EEG, and annotation data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import pyedflib
import mne
from pathlib import Path
import logging

from .data_structures import (
    EMGSignal, EEGSignal, RespiratoryEvent, ArousalEvent, 
    SleepStage, SleepStageSegment, AnalysisConfig
)

logger = logging.getLogger(__name__)


class SleepDataLoader:
    """Main class for loading sleep study data from various formats."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize the data loader with optional configuration."""
        self.config = config or AnalysisConfig()
        self.supported_formats = ['.edf', '.bdf', '.gdf']
    
    def load_edf_file(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Load data from an EDF file.
        
        Args:
            file_path: Path to the EDF file
            
        Returns:
            Dictionary containing loaded signals and metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"EDF file not found: {file_path}")
        
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        logger.info(f"Loading EDF file: {file_path}")
        
        try:
            # Load with MNE for better handling
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            
            # Extract basic information
            info = {
                'file_path': str(file_path),
                'duration': raw.times[-1],
                'sampling_frequency': raw.info['sfreq'],
                'channels': raw.ch_names,
                'start_time': raw.info['meas_date'],
                'signals': {},
                'annotations': raw.annotations if hasattr(raw, 'annotations') else []
            }
            
            # Extract EMG channels
            emg_channels = self._identify_emg_channels(raw.ch_names)
            for ch_name in emg_channels:
                ch_data = raw.get_data(picks=ch_name)[0]
                info['signals'][ch_name] = EMGSignal(
                    data=ch_data,
                    sampling_frequency=raw.info['sfreq'],
                    channel_name=ch_name
                )
                logger.info(f"Loaded EMG channel: {ch_name}")
            
            # Extract EEG channels
            eeg_channels = self._identify_eeg_channels(raw.ch_names)
            for ch_name in eeg_channels:
                ch_data = raw.get_data(picks=ch_name)[0]
                info['signals'][ch_name] = EEGSignal(
                    data=ch_data,
                    sampling_frequency=raw.info['sfreq'],
                    channel_name=ch_name
                )
                logger.info(f"Loaded EEG channel: {ch_name}")
            
            # Parse annotations for events
            if raw.annotations:
                info['respiratory_events'] = self._parse_respiratory_events(raw.annotations)
                info['arousal_events'] = self._parse_arousal_events(raw.annotations)
                info['sleep_stages'] = self._parse_sleep_stages(raw.annotations)
            
            logger.info(f"Successfully loaded {len(info['signals'])} channels")
            return info
            
        except Exception as e:
            logger.error(f"Error loading EDF file: {e}")
            raise
    
    def _identify_emg_channels(self, channel_names: List[str]) -> List[str]:
        """Identify EMG channels based on naming conventions."""
        emg_keywords = ['emg', 'tib', 'tibialis', 'leg', 'ant', 'anterior']
        emg_channels = []
        
        for ch_name in channel_names:
            ch_lower = ch_name.lower()
            if any(keyword in ch_lower for keyword in emg_keywords):
                emg_channels.append(ch_name)
        
        return emg_channels
    
    def _identify_eeg_channels(self, channel_names: List[str]) -> List[str]:
        """Identify EEG channels based on naming conventions."""
        eeg_keywords = ['eeg', 'f3', 'f4', 'c3', 'c4', 'o1', 'o2', 'm1', 'm2']
        eeg_channels = []
        
        for ch_name in channel_names:
            ch_lower = ch_name.lower()
            if any(keyword in ch_lower for keyword in eeg_keywords):
                eeg_channels.append(ch_name)
        
        return eeg_channels
    
    def _parse_respiratory_events(self, annotations: List) -> List[RespiratoryEvent]:
        """Parse respiratory events from annotations."""
        respiratory_events = []
        resp_keywords = ['apnea', 'hypopnea', 'resp', 'breathing']
        
        for ann in annotations:
            description = ann['description'].lower()
            if any(keyword in description for keyword in resp_keywords):
                event = RespiratoryEvent(
                    start_time=ann['onset'],
                    end_time=ann['onset'] + ann['duration'],
                    event_type=description,
                    duration=ann['duration']
                )
                respiratory_events.append(event)
        
        return respiratory_events
    
    def _parse_arousal_events(self, annotations: List) -> List[ArousalEvent]:
        """Parse arousal events from annotations."""
        arousal_events = []
        arousal_keywords = ['arousal', 'awake', 'wake']
        
        for ann in annotations:
            description = ann['description'].lower()
            if any(keyword in description for keyword in arousal_keywords):
                event = ArousalEvent(
                    start_time=ann['onset'],
                    end_time=ann['onset'] + ann['duration'],
                    duration=ann['duration']
                )
                arousal_events.append(event)
        
        return arousal_events
    
    def _parse_sleep_stages(self, annotations: List) -> List[SleepStageSegment]:
        """Parse sleep stages from annotations."""
        sleep_stages = []
        stage_mapping = {
            'wake': SleepStage.WAKE,
            'n1': SleepStage.N1,
            'n2': SleepStage.N2,
            'n3': SleepStage.N3,
            'rem': SleepStage.REM,
            'stage 1': SleepStage.N1,
            'stage 2': SleepStage.N2,
            'stage 3': SleepStage.N3,
            'stage w': SleepStage.WAKE
        }
        
        for ann in annotations:
            description = ann['description'].lower()
            stage = None
            
            for key, value in stage_mapping.items():
                if key in description:
                    stage = value
                    break
            
            if stage is None:
                stage = SleepStage.UNKNOWN
            
            segment = SleepStageSegment(
                stage=stage,
                start_time=ann['onset'],
                end_time=ann['onset'] + ann['duration'],
                duration=ann['duration']
            )
            sleep_stages.append(segment)
        
        return sleep_stages
    
    def load_annotations_file(self, file_path: Union[str, Path]) -> Dict[str, any]:
        """
        Load annotations from a separate file (CSV, JSON, etc.).
        
        Args:
            file_path: Path to the annotations file
            
        Returns:
            Dictionary containing parsed events
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Annotations file not found: {file_path}")
        
        logger.info(f"Loading annotations file: {file_path}")
        
        if file_path.suffix.lower() == '.csv':
            return self._load_csv_annotations(file_path)
        elif file_path.suffix.lower() == '.json':
            return self._load_json_annotations(file_path)
        else:
            raise ValueError(f"Unsupported annotations format: {file_path.suffix}")
    
    def _load_csv_annotations(self, file_path: Path) -> Dict[str, any]:
        """Load annotations from CSV file."""
        try:
            df = pd.read_csv(file_path)
            events = {
                'respiratory_events': [],
                'arousal_events': [],
                'sleep_stages': []
            }
            
            # Parse based on column names
            if 'start_time' in df.columns and 'end_time' in df.columns:
                for _, row in df.iterrows():
                    if 'type' in df.columns:
                        event_type = row['type'].lower()
                        
                        if 'apnea' in event_type or 'hypopnea' in event_type:
                            event = RespiratoryEvent(
                                start_time=row['start_time'],
                                end_time=row['end_time'],
                                event_type=event_type,
                                duration=row['end_time'] - row['start_time']
                            )
                            events['respiratory_events'].append(event)
                        
                        elif 'arousal' in event_type:
                            event = ArousalEvent(
                                start_time=row['start_time'],
                                end_time=row['end_time'],
                                duration=row['end_time'] - row['start_time']
                            )
                            events['arousal_events'].append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error loading CSV annotations: {e}")
            raise
    
    def _load_json_annotations(self, file_path: Path) -> Dict[str, any]:
        """Load annotations from JSON file."""
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            events = {
                'respiratory_events': [],
                'arousal_events': [],
                'sleep_stages': []
            }
            
            # Parse based on JSON structure
            if 'respiratory_events' in data:
                for event_data in data['respiratory_events']:
                    event = RespiratoryEvent(**event_data)
                    events['respiratory_events'].append(event)
            
            if 'arousal_events' in data:
                for event_data in data['arousal_events']:
                    event = ArousalEvent(**event_data)
                    events['arousal_events'].append(event)
            
            return events
            
        except Exception as e:
            logger.error(f"Error loading JSON annotations: {e}")
            raise
    
    def validate_data_quality(self, signals: Dict[str, any]) -> Dict[str, float]:
        """
        Validate the quality of loaded signals.
        
        Args:
            signals: Dictionary of loaded signals
            
        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}
        
        for ch_name, signal in signals.items():
            if isinstance(signal, (EMGSignal, EEGSignal)):
                # Calculate signal-to-noise ratio
                signal_power = np.mean(signal.data ** 2)
                noise_power = np.var(signal.data)
                snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 100
                
                # Calculate baseline stability
                baseline_std = np.std(signal.data)
                baseline_stability = 1.0 / (1.0 + baseline_std)
                
                # Calculate artifact level (simplified)
                threshold = np.mean(signal.data) + 3 * np.std(signal.data)
                artifact_samples = np.sum(np.abs(signal.data) > threshold)
                artifact_level = artifact_samples / len(signal.data)
                
                quality_metrics[ch_name] = {
                    'snr': snr,
                    'baseline_stability': baseline_stability,
                    'artifact_level': artifact_level,
                    'overall_quality': (snr/100 + baseline_stability + (1-artifact_level)) / 3
                }
        
        return quality_metrics


def load_sleep_study(file_path: Union[str, Path], 
                    annotations_path: Optional[Union[str, Path]] = None,
                    config: Optional[AnalysisConfig] = None) -> Dict[str, any]:
    """
    Convenience function to load a complete sleep study.
    
    Args:
        file_path: Path to the main data file (EDF, etc.)
        annotations_path: Optional path to separate annotations file
        config: Optional analysis configuration
        
    Returns:
        Complete sleep study data dictionary
    """
    loader = SleepDataLoader(config)
    
    # Load main data file
    study_data = loader.load_edf_file(file_path)
    
    # Load annotations if provided
    if annotations_path:
        annotations = loader.load_annotations_file(annotations_path)
        study_data.update(annotations)
    
    # Validate data quality
    study_data['quality_metrics'] = loader.validate_data_quality(study_data['signals'])
    
    return study_data 