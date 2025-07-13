"""
Analysis module for WASM 2019 leg movement detection software.

This module implements:
- STEP 7: Respiratory association (CLMr)
- STEP 8: Arousal association (PLMSa)
- STEP 9: Sleep stage segmentation
- STEP 10: Metrics calculation
"""

from .association_analysis import (
    associate_with_respiratory,
    associate_with_arousals,
    split_by_sleep_stage
)

from .metrics_calculation import (
    compute_analysis_metrics,
    calculate_plms_index,
    calculate_periodicity_index,
    generate_diagnostic_result
)

__all__ = [
    'associate_with_respiratory',
    'associate_with_arousals',
    'split_by_sleep_stage',
    'compute_analysis_metrics',
    'calculate_plms_index',
    'calculate_periodicity_index',
    'generate_diagnostic_result'
] 