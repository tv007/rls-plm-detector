#!/usr/bin/env python3
"""
Extract metadata from all EDF files in the data directory and save as a CSV summary.
"""

import os
from pathlib import Path
import pandas as pd
import mne

DATA_DIR = Path("data")
OUTPUT_DIR = Path("examples/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_CSV = OUTPUT_DIR / "edf_metadata_summary.csv"

# Find all EDF files in the data directory
edf_files = list(DATA_DIR.glob("*.edf"))

metadata_rows = []

for edf_path in edf_files:
    try:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
        info = raw.info
        channels = info["ch_names"]
        sfreq = info["sfreq"]
        meas_date = info["meas_date"]
        duration = raw.times[-1] if hasattr(raw, "times") else None
        n_channels = len(channels)
        
        # For each channel, get type and sampling rate
        for idx, ch_name in enumerate(channels):
            ch_type = info["chs"][idx]["kind"] if "chs" in info and idx < len(info["chs"]) else "unknown"
            ch_type_str = mne.io.pick.channel_type(info, idx) if hasattr(mne.io.pick, "channel_type") else "unknown"
            ch_sfreq = info["sfreq"] if "sfreq" in info else "unknown"
            metadata_rows.append({
                "edf_file": edf_path.name,
                "channel_index": idx,
                "channel_name": ch_name,
                "channel_type": ch_type_str,
                "sampling_frequency": ch_sfreq,
                "n_channels": n_channels,
                "duration_sec": duration,
                "meas_date": str(meas_date) if meas_date else "unknown"
            })
    except Exception as e:
        print(f"Failed to read {edf_path}: {e}")

# Create DataFrame and save as CSV
meta_df = pd.DataFrame(metadata_rows)
meta_df.to_csv(OUTPUT_CSV, index=False)
print(f"Metadata summary saved to: {OUTPUT_CSV}") 