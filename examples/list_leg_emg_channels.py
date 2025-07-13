#!/usr/bin/env python3
"""
List all PLMl and PLMr channels from EDF metadata that are used for leg movement detection.
This script helps you understand exactly which signals are used by the algorithm.
"""

import pandas as pd
from pathlib import Path

META_CSV = Path("examples/output/edf_metadata_summary.csv")
OUTPUT_CSV = Path("examples/output/edf_leg_emg_channels.csv")

# Only use PLMl and PLMr for leg movement detection
leg_channel_names = ['plml', 'plmr']

# Read the metadata CSV
meta_df = pd.read_csv(META_CSV)

# Find channels matching exactly PLMl or PLMr (case-insensitive)
mask = meta_df['channel_name'].str.lower().isin(leg_channel_names)
leg_emg_df = meta_df[mask].copy()

# Save filtered channels to a new CSV
leg_emg_df.to_csv(OUTPUT_CSV, index=False)

# Print a summary to the console
if not leg_emg_df.empty:
    print("\nLeg EMG channels used for leg movement detection (STRICT: only PLMl and PLMr):")
    print(leg_emg_df[['edf_file', 'channel_index', 'channel_name', 'sampling_frequency']].to_string(index=False))
    print(f"\nTotal: {len(leg_emg_df)} channels found across {leg_emg_df['edf_file'].nunique()} EDF files.")
    print(f"\nDetailed list saved to: {OUTPUT_CSV}")
else:
    print("No PLMl or PLMr channels found in the metadata. Check your EDF files and channel naming conventions.")

print("\nThe algorithm now uses ONLY these channels (PLMl and PLMr) for leg movement detection.\n")
print("If you want to change which channels are used, modify the 'leg_channel_names' list in this script.") 