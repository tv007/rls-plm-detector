"""
Main analysis pipeline for WASM 2019 leg movement detection software.

This module integrates all steps of the WASM 2019 algorithm into a complete workflow:
1. Data loading and preprocessing
2. Baseline estimation
3. Leg movement detection
4. Association analysis
5. Metrics calculation
6. Diagnostic reporting
"""

import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

from .utils.data_structures import (
    AnalysisConfig, DiagnosticResult, EMGSignal, SleepStageSegment
)
from .utils.data_loader import load_sleep_study
from .preprocessing.signal_processing import preprocess_emg
from .preprocessing.baseline_estimation import compute_baseline
from .detection.leg_movement_detection import (
    detect_leg_movements, filter_candidate_leg_movements,
    combine_bilateral_clms, detect_plm_sequences
)
from .analysis.association_analysis import (
    associate_with_respiratory, associate_with_arousals, split_by_sleep_stage
)
from .analysis.metrics_calculation import (
    compute_analysis_metrics, generate_diagnostic_result
)

logger = logging.getLogger(__name__)


class WASMAnalyzer:
    """
    Main analyzer class for WASM 2019 leg movement detection.
    
    This class orchestrates the complete analysis pipeline from raw data
    to diagnostic results.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        """
        Initialize the WASM analyzer.
        
        Args:
            config: Analysis configuration parameters
        """
        self.config = config or AnalysisConfig()
        self._validate_config()
        
        # Initialize logging
        self._setup_logging()
        
        logger.info("Initialized WASM 2019 analyzer")
    
    def _validate_config(self):
        """Validate the analysis configuration."""
        from .preprocessing.signal_processing import validate_preprocessing_parameters
        from .preprocessing.baseline_estimation import validate_baseline_parameters
        from .detection.leg_movement_detection import validate_detection_parameters
        from .analysis.association_analysis import validate_association_parameters
        from .analysis.metrics_calculation import validate_metrics_parameters
        
        validators = [
            validate_preprocessing_parameters,
            validate_baseline_parameters,
            validate_detection_parameters,
            validate_association_parameters,
            validate_metrics_parameters
        ]
        
        for validator in validators:
            if not validator(self.config):
                raise ValueError("Configuration validation failed")
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def analyze_sleep_study(self, 
                          file_path: str,
                          annotations_path: Optional[str] = None,
                          patient_id: str = "unknown",
                          recording_date: str = "") -> DiagnosticResult:
        """
        Perform complete analysis of a sleep study.
        
        Args:
            file_path: Path to the sleep study file (EDF, etc.)
            annotations_path: Optional path to annotations file
            patient_id: Patient identifier
            recording_date: Recording date
            
        Returns:
            Complete diagnostic result
        """
        logger.info(f"Starting analysis for patient: {patient_id}")
        
        try:
            # Step 1: Load and preprocess data
            study_data = self._load_and_preprocess_data(file_path, annotations_path)
            
            # Step 2: Analyze each EMG channel
            all_clms = []
            all_plm_sequences = []
            
            for channel_name, signal in study_data['signals'].items():
                if isinstance(signal, EMGSignal):
                    logger.info(f"Analyzing EMG channel: {channel_name}")
                    
                    # Preprocess EMG signal
                    preprocessed_signal = preprocess_emg(signal, self.config)
                    
                    # Compute baseline
                    baseline = compute_baseline(preprocessed_signal, self.config)
                    
                    # Detect leg movements
                    leg_movements = detect_leg_movements(preprocessed_signal, baseline, self.config)
                    
                    # Filter to candidate leg movements
                    clms = filter_candidate_leg_movements(leg_movements, self.config)
                    
                    all_clms.extend(clms)
            
            # Step 3: Combine bilateral movements
            if len(all_clms) > 1:
                left_clms = [clm for clm in all_clms if clm.side == 'left']
                right_clms = [clm for clm in all_clms if clm.side == 'right']
                
                if left_clms and right_clms:
                    bilateral_clms = combine_bilateral_clms(left_clms, right_clms, self.config)
                    all_clms.extend(bilateral_clms)
            
            # Step 4: Detect PLM sequences
            all_plm_sequences = detect_plm_sequences(all_clms, self.config)
            
            # Step 5: Association analysis
            if 'respiratory_events' in study_data:
                all_clms = associate_with_respiratory(
                    all_clms, study_data['respiratory_events'], self.config
                )
            
            if 'arousal_events' in study_data:
                all_clms = associate_with_arousals(
                    all_clms, study_data['arousal_events'], self.config
                )
            
            # Step 6: Sleep stage analysis
            sleep_stages = study_data.get('sleep_stages', [])
            if sleep_stages:
                clms_by_stage = split_by_sleep_stage(all_clms, sleep_stages)
                logger.info("Completed sleep stage analysis")
            
            # Step 7: Calculate metrics
            metrics = compute_analysis_metrics(
                all_clms, all_plm_sequences, sleep_stages, self.config
            )
            
            # Step 8: Generate diagnostic result
            diagnostic_result = generate_diagnostic_result(
                metrics, all_clms, all_plm_sequences, sleep_stages,
                patient_id, recording_date, self.config
            )
            
            logger.info(f"Completed analysis for patient: {patient_id}")
            logger.info(f"Diagnosis: {diagnostic_result.plms_diagnosis}")
            
            return diagnostic_result
            
        except Exception as e:
            logger.error(f"Analysis failed for patient {patient_id}: {e}")
            raise
    
    def _load_and_preprocess_data(self, file_path: str, 
                                 annotations_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load and preprocess sleep study data.
        
        Args:
            file_path: Path to the sleep study file
            annotations_path: Optional path to annotations file
            
        Returns:
            Preprocessed study data
        """
        logger.info(f"Loading sleep study data from: {file_path}")
        
        # Load data
        study_data = load_sleep_study(file_path, annotations_path, self.config)
        
        # Validate data quality
        if 'quality_metrics' in study_data:
            for channel, quality in study_data['quality_metrics'].items():
                if quality['overall_quality'] < 0.5:
                    logger.warning(f"Low quality signal detected for channel: {channel}")
        
        logger.info("Data loading and preprocessing completed")
        return study_data
    
    def analyze_batch(self, 
                     file_paths: List[str],
                     patient_ids: Optional[List[str]] = None,
                     output_dir: Optional[str] = None) -> List[DiagnosticResult]:
        """
        Analyze multiple sleep studies in batch.
        
        Args:
            file_paths: List of file paths to analyze
            patient_ids: Optional list of patient IDs
            output_dir: Optional output directory for results
            
        Returns:
            List of diagnostic results
        """
        logger.info(f"Starting batch analysis of {len(file_paths)} files")
        
        results = []
        
        for i, file_path in enumerate(file_paths):
            patient_id = patient_ids[i] if patient_ids and i < len(patient_ids) else f"patient_{i+1}"
            
            try:
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                result = self.analyze_sleep_study(file_path, patient_id=patient_id)
                results.append(result)
                
                # Save result if output directory is specified
                if output_dir:
                    self._save_result(result, output_dir, patient_id)
                
            except Exception as e:
                logger.error(f"Failed to analyze {file_path}: {e}")
                # Continue with next file
        
        logger.info(f"Completed batch analysis. Successfully processed {len(results)}/{len(file_paths)} files")
        return results
    
    def _save_result(self, result: DiagnosticResult, output_dir: str, patient_id: str):
        """
        Save analysis result to file.
        
        Args:
            result: Diagnostic result
            output_dir: Output directory
            patient_id: Patient identifier
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        import json
        from datetime import datetime
        
        result_dict = {
            'patient_id': result.patient_id,
            'recording_date': result.recording_date,
            'analysis_date': result.analysis_date,
            'plms_diagnosis': result.plms_diagnosis,
            'clinical_significance': result.clinical_significance,
            'recommendations': result.recommendations,
            'metrics': {
                'plms_index': result.metrics.plms_index,
                'plmw_index': result.metrics.plmw_index,
                'total_clms': result.metrics.total_clms,
                'total_plms': result.metrics.total_plms,
                'periodicity_index': result.metrics.periodicity_index,
                'respiratory_association_index': result.metrics.respiratory_association_index,
                'arousal_association_index': result.metrics.arousal_association_index
            }
        }
        
        json_path = output_path / f"{patient_id}_analysis.json"
        with open(json_path, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        logger.info(f"Saved result to: {json_path}")


def analyze_sleep_study(file_path: str,
                       config: Optional[AnalysisConfig] = None,
                       **kwargs) -> DiagnosticResult:
    """
    Convenience function for analyzing a single sleep study.
    
    Args:
        file_path: Path to the sleep study file
        config: Optional analysis configuration
        **kwargs: Additional arguments passed to analyze_sleep_study
        
    Returns:
        Diagnostic result
    """
    analyzer = WASMAnalyzer(config)
    return analyzer.analyze_sleep_study(file_path, **kwargs)


def analyze_batch(file_paths: List[str],
                 config: Optional[AnalysisConfig] = None,
                 **kwargs) -> List[DiagnosticResult]:
    """
    Convenience function for batch analysis.
    
    Args:
        file_paths: List of file paths to analyze
        config: Optional analysis configuration
        **kwargs: Additional arguments passed to analyze_batch
        
    Returns:
        List of diagnostic results
    """
    analyzer = WASMAnalyzer(config)
    return analyzer.analyze_batch(file_paths, **kwargs) 