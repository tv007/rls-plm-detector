"""
Basic tests for WASM 2019 leg movement detection software.

These tests verify that the core components can be imported and basic
functionality works as expected.
"""

import unittest
import numpy as np
from pathlib import Path

# Import core modules
from src.utils.data_structures import (
    AnalysisConfig, EMGSignal, CandidateLegMovement, PLMSequence
)
from src.preprocessing.signal_processing import preprocess_emg
from src.preprocessing.baseline_estimation import compute_baseline
from src.detection.leg_movement_detection import detect_leg_movements
from src.analysis.metrics_calculation import calculate_plms_index


class TestBasicFunctionality(unittest.TestCase):
    """Test basic functionality of the WASM 2019 software."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = AnalysisConfig()
        
        # Create a simple test EMG signal
        fs = 1000  # 1 kHz sampling frequency
        duration = 10  # 10 seconds
        t = np.linspace(0, duration, int(fs * duration))
        
        # Create synthetic EMG with some "movements"
        self.test_signal = np.random.normal(0, 1, len(t))
        
        # Add some simulated leg movements
        movement_times = [2, 4, 6, 8]  # seconds
        for time in movement_times:
            start_idx = int(time * fs)
            end_idx = start_idx + int(0.5 * fs)  # 0.5 second movement
            self.test_signal[start_idx:end_idx] += 10  # Add amplitude
        
        self.emg_signal = EMGSignal(
            data=self.test_signal,
            sampling_frequency=fs,
            channel_name="test_emg"
        )
    
    def test_config_creation(self):
        """Test that configuration can be created."""
        config = AnalysisConfig()
        self.assertIsNotNone(config)
        self.assertEqual(config.emg_bandpass_low, 10.0)
        self.assertEqual(config.emg_bandpass_high, 100.0)
    
    def test_emg_signal_creation(self):
        """Test that EMG signal can be created."""
        self.assertIsNotNone(self.emg_signal)
        self.assertEqual(len(self.emg_signal.data), 10000)  # 10 seconds at 1kHz
        self.assertEqual(self.emg_signal.sampling_frequency, 1000)
    
    def test_preprocessing(self):
        """Test that preprocessing works."""
        preprocessed = preprocess_emg(self.emg_signal, self.config)
        self.assertIsNotNone(preprocessed)
        self.assertEqual(len(preprocessed.data), len(self.emg_signal.data))
    
    def test_baseline_estimation(self):
        """Test that baseline estimation works."""
        preprocessed = preprocess_emg(self.emg_signal, self.config)
        baseline = compute_baseline(preprocessed, self.config)
        self.assertIsNotNone(baseline)
        self.assertEqual(len(baseline), len(preprocessed.data))
    
    def test_leg_movement_detection(self):
        """Test that leg movement detection works."""
        preprocessed = preprocess_emg(self.emg_signal, self.config)
        baseline = compute_baseline(preprocessed, self.config)
        movements = detect_leg_movements(preprocessed, baseline, self.config)
        self.assertIsNotNone(movements)
        self.assertIsInstance(movements, list)
    
    def test_plms_index_calculation(self):
        """Test that PLMS index calculation works."""
        # Create a simple PLM sequence
        movements = [
            CandidateLegMovement(
                start_time=1.0, end_time=1.5, duration=0.5,
                amplitude=10.0, baseline=1.0, channel="test", side="left"
            ),
            CandidateLegMovement(
                start_time=2.0, end_time=2.5, duration=0.5,
                amplitude=10.0, baseline=1.0, channel="test", side="left"
            )
        ]
        
        plm_sequence = PLMSequence(
            movements=movements,
            start_time=1.0, end_time=2.5, duration=1.5,
            movement_count=2, mean_imi=1.0, imi_std=0.0,
            periodicity_index=2.0
        )
        
        total_sleep_time = 3600  # 1 hour
        plms_index = calculate_plms_index([plm_sequence], total_sleep_time)
        self.assertEqual(plms_index, 2.0)  # 2 PLMs per hour
    
    def test_imports(self):
        """Test that all core modules can be imported."""
        try:
            from src.main_analysis import WASMAnalyzer
            from src.utils.data_loader import SleepDataLoader
            from src.analysis.association_analysis import associate_with_respiratory
            from src.analysis.metrics_calculation import compute_analysis_metrics
        except ImportError as e:
            self.fail(f"Failed to import core modules: {e}")


class TestDataStructures(unittest.TestCase):
    """Test data structure functionality."""
    
    def test_candidate_leg_movement(self):
        """Test CandidateLegMovement creation."""
        clm = CandidateLegMovement(
            start_time=1.0, end_time=1.5, duration=0.5,
            amplitude=10.0, baseline=1.0, channel="test", side="left"
        )
        self.assertEqual(clm.duration, 0.5)
        self.assertEqual(clm.side, "left")
    
    def test_plm_sequence(self):
        """Test PLMSequence creation."""
        movements = [
            CandidateLegMovement(
                start_time=1.0, end_time=1.5, duration=0.5,
                amplitude=10.0, baseline=1.0, channel="test", side="left"
            )
        ]
        
        sequence = PLMSequence(
            movements=movements,
            start_time=1.0, end_time=1.5, duration=0.5,
            movement_count=1, mean_imi=0.0, imi_std=0.0,
            periodicity_index=1.0
        )
        
        self.assertEqual(sequence.movement_count, 1)
        self.assertEqual(len(sequence.movements), 1)


if __name__ == '__main__':
    unittest.main() 