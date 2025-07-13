#!/usr/bin/env python3
"""
Basic usage example for WASM 2019 leg movement detection software.

This example demonstrates how to:
1. Load and analyze a sleep study
2. Configure analysis parameters
3. Generate diagnostic results
4. Save and display results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os

# Add the parent directory to the path to import the software
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.main_analysis import analyze_sleep_study
from src.utils.data_structures import AnalysisConfig, EMGSignal
from src.preprocessing.signal_processing import preprocess_emg
from src.preprocessing.baseline_estimation import compute_baseline
from src.detection.leg_movement_detection import detect_leg_movements, filter_candidate_leg_movements


def create_synthetic_data():
    """
    Create synthetic EMG data for demonstration purposes.
    
    In a real application, this would be replaced with actual EDF file loading.
    """
    print("Creating synthetic EMG data...")
    
    # Parameters
    fs = 1000  # 1 kHz sampling frequency
    duration = 60  # 60 seconds
    t = np.linspace(0, duration, int(fs * duration))
    
    # Create baseline EMG signal
    emg_signal = np.random.normal(0, 0.5, len(t))
    
    # Add some simulated leg movements
    movement_times = [10, 20, 30, 40, 50]  # seconds
    for time in movement_times:
        start_idx = int(time * fs)
        end_idx = start_idx + int(0.8 * fs)  # 0.8 second movement
        emg_signal[start_idx:end_idx] += 8 + np.random.normal(0, 1, end_idx - start_idx)
    
    # Create EMG signal object
    emg = EMGSignal(
        data=emg_signal,
        sampling_frequency=fs,
        channel_name="synthetic_emg"
    )
    
    return emg


def demonstrate_analysis():
    """Demonstrate the complete analysis pipeline."""
    print("="*60)
    print("WASM 2019 Leg Movement Detection - Basic Usage Example")
    print("="*60)
    
    # Step 1: Create synthetic data
    emg_signal = create_synthetic_data()
    
    # Step 2: Configure analysis parameters
    config = AnalysisConfig(
        emg_bandpass_low=10.0,
        emg_bandpass_high=100.0,
        smoothing_window_ms=300.0,
        amplitude_threshold=8.0,
        minimum_duration=0.5,
        maximum_duration=10.0,
        min_imi=10.0,
        max_imi=90.0,
        min_plm_sequence_length=4
    )
    
    print(f"Analysis configuration:")
    print(f"  - Bandpass filter: {config.emg_bandpass_low}-{config.emg_bandpass_high} Hz")
    print(f"  - Smoothing window: {config.smoothing_window_ms} ms")
    print(f"  - Amplitude threshold: {config.amplitude_threshold} µV")
    print(f"  - Duration range: {config.minimum_duration}-{config.maximum_duration} s")
    print(f"  - IMI range: {config.min_imi}-{config.max_imi} s")
    
    # Step 3: Preprocess the signal
    print("\nStep 1: Preprocessing EMG signal...")
    preprocessed = preprocess_emg(emg_signal, config)
    print(f"  - Signal length: {len(preprocessed.data)} samples")
    print(f"  - Duration: {preprocessed.duration:.1f} seconds")
    
    # Step 4: Compute baseline
    print("\nStep 2: Computing baseline...")
    baseline = compute_baseline(preprocessed, config)
    print(f"  - Baseline computed for {len(baseline)} samples")
    
    # Step 5: Detect leg movements
    print("\nStep 3: Detecting leg movements...")
    leg_movements = detect_leg_movements(preprocessed, baseline, config)
    print(f"  - Detected {len(leg_movements)} leg movements")
    
    # Step 6: Filter to candidate leg movements
    print("\nStep 4: Filtering candidate leg movements...")
    clms = filter_candidate_leg_movements(leg_movements, config)
    print(f"  - Filtered to {len(clms)} candidate leg movements")
    
    # Step 7: Display movement details
    if clms:
        print("\nDetected movements:")
        for i, clm in enumerate(clms[:5]):  # Show first 5
            print(f"  Movement {i+1}: {clm.duration:.2f}s, amplitude: {clm.amplitude:.1f}µV")
        if len(clms) > 5:
            print(f"  ... and {len(clms) - 5} more movements")
    
    # Step 8: Create a simple visualization
    print("\nStep 5: Creating visualization...")
    create_visualization(emg_signal, preprocessed, baseline, clms)
    
    print("\nAnalysis completed successfully!")
    print("="*60)


def create_visualization(original_signal, preprocessed_signal, baseline, clms):
    """Create a simple visualization of the analysis results."""
    try:
        # Create figure with subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))
        fig.suptitle('WASM 2019 Leg Movement Detection Analysis', fontsize=16)
        
        # Plot 1: Original signal
        time_axis = original_signal.time_axis
        axes[0].plot(time_axis, original_signal.data, 'b-', linewidth=0.5, alpha=0.7)
        axes[0].set_title('Original EMG Signal')
        axes[0].set_ylabel('Amplitude (µV)')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Preprocessed signal with baseline
        axes[1].plot(time_axis, preprocessed_signal.data, 'g-', linewidth=0.5, alpha=0.7)
        axes[1].plot(time_axis, baseline, 'r-', linewidth=1, alpha=0.8, label='Baseline')
        axes[1].set_title('Preprocessed Signal with Baseline')
        axes[1].set_ylabel('Amplitude (µV)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Detected movements
        axes[2].plot(time_axis, preprocessed_signal.data, 'g-', linewidth=0.5, alpha=0.7)
        axes[2].plot(time_axis, baseline, 'r-', linewidth=1, alpha=0.8, label='Baseline')
        
        # Mark detected movements
        for clm in clms:
            start_idx = int(clm.start_time * original_signal.sampling_frequency)
            end_idx = int(clm.end_time * original_signal.sampling_frequency)
            axes[2].axvspan(clm.start_time, clm.end_time, alpha=0.3, color='yellow', label='Detected Movement' if clm == clms[0] else "")
        
        axes[2].set_title(f'Detected Leg Movements ({len(clms)} total)')
        axes[2].set_xlabel('Time (seconds)')
        axes[2].set_ylabel('Amplitude (µV)')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Adjust layout and save
        plt.tight_layout()
        
        # Create output directory
        output_dir = Path("examples/output")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save plot
        plot_path = output_dir / "analysis_visualization.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"  - Visualization saved to: {plot_path}")
        
        # Show plot (optional - comment out if running in headless environment)
        # plt.show()
        
    except ImportError:
        print("  - Matplotlib not available, skipping visualization")
    except Exception as e:
        print(f"  - Visualization failed: {e}")


def demonstrate_configuration():
    """Demonstrate different configuration options."""
    print("\n" + "="*60)
    print("Configuration Examples")
    print("="*60)
    
    # Default configuration
    default_config = AnalysisConfig()
    print("Default configuration:")
    print(f"  - Amplitude threshold: {default_config.amplitude_threshold} µV")
    print(f"  - Duration range: {default_config.minimum_duration}-{default_config.maximum_duration} s")
    
    # Custom configuration for sensitive detection
    sensitive_config = AnalysisConfig(
        amplitude_threshold=5.0,  # Lower threshold
        minimum_duration=0.3,     # Shorter minimum duration
        max_imi=120.0            # Longer IMI range
    )
    print("\nSensitive configuration:")
    print(f"  - Amplitude threshold: {sensitive_config.amplitude_threshold} µV")
    print(f"  - Duration range: {sensitive_config.minimum_duration}-{sensitive_config.maximum_duration} s")
    print(f"  - Max IMI: {sensitive_config.max_imi} s")
    
    # Custom configuration for conservative detection
    conservative_config = AnalysisConfig(
        amplitude_threshold=12.0,  # Higher threshold
        minimum_duration=0.8,      # Longer minimum duration
        min_plm_sequence_length=5  # More movements required for PLM sequence
    )
    print("\nConservative configuration:")
    print(f"  - Amplitude threshold: {conservative_config.amplitude_threshold} µV")
    print(f"  - Duration range: {conservative_config.minimum_duration}-{conservative_config.maximum_duration} s")
    print(f"  - Min PLM sequence length: {conservative_config.min_plm_sequence_length}")


if __name__ == "__main__":
    # Run the demonstration
    demonstrate_analysis()
    demonstrate_configuration()
    
    print("\n" + "="*60)
    print("Example completed successfully!")
    print("Check the 'examples/output' directory for generated files.")
    print("="*60) 