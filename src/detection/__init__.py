"""
Detection module for WASM 2019 leg movement detection software.

This module implements the core detection algorithms:
- STEP 3: Leg movement detection
- STEP 4: Candidate leg movement filtering
- STEP 5: Bilateral CLM combination
- STEP 6: Periodic leg movement sequence detection
"""

from .leg_movement_detection import (
    detect_leg_movements,
    filter_candidate_leg_movements,
    combine_bilateral_clms,
    detect_plm_sequences
)

__all__ = [
    'detect_leg_movements',
    'filter_candidate_leg_movements',
    'combine_bilateral_clms',
    'detect_plm_sequences'
] 