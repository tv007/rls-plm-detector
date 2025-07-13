#!/usr/bin/env python3
"""
Visualize PLMl and PLMr EMG signals from the first EDF file in the data directory as a clinical time series dashboard.
Efficiently handles large signals and provides clinical-grade plots.
"""

import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Settings
DATA_DIR = Path("data")
OUTPUT_DIR = Path("examples/output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DASHBOARD_TITLE = "Leg Movement EMG Clinical Dashboard"
PLOT_DURATION_SEC = 60  # Duration of each window to plot (seconds)
DOWNSAMPLE_FACTOR = 10  # Downsample for plotting if needed

# Find the first EDF file in the data directory
edf_files = sorted(DATA_DIR.glob("*.edf"))
if not edf_files:
    print("No EDF files found in the data directory.")
    exit(1)
edf_path = edf_files[0]
print(f"Visualizing file: {edf_path.name}")

try:
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    ch_names = [ch.lower() for ch in raw.ch_names]
    plml_idx = next((i for i, ch in enumerate(ch_names) if ch == 'plml'), None)
    plmr_idx = next((i for i, ch in enumerate(ch_names) if ch == 'plmr'), None)
    if plml_idx is None or plmr_idx is None:
        print(f"PLMl or PLMr channel not found in {edf_path.name}")
        exit(1)
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    duration_sec = n_samples / sfreq
    print(f"  Duration: {duration_sec/3600:.2f} hours, Sampling rate: {sfreq} Hz")
    # Efficiently read only the PLMl and PLMr channels
    data, times = raw[[plml_idx, plmr_idx], :]
    # Downsample for visualization if needed
    if sfreq > 100:
        data = data[:, ::DOWNSAMPLE_FACTOR]
        times = times[::DOWNSAMPLE_FACTOR]
        sfreq = sfreq / DOWNSAMPLE_FACTOR
    # Plot in windows for large signals
    window_samples = int(PLOT_DURATION_SEC * sfreq)
    n_windows = int(np.ceil(data.shape[1] / window_samples))
    for w in range(n_windows):
        start = w * window_samples
        end = min((w + 1) * window_samples, data.shape[1])
        if end - start < 2:
            continue
        fig, ax = plt.subplots(2, 1, figsize=(15, 6), sharex=True)
        fig.suptitle(f"{DASHBOARD_TITLE}\nFile: {edf_path.name} | Window: {w+1}/{n_windows}", fontsize=16)
        # PLMl
        ax[0].plot(times[start:end], data[0, start:end], color='b', lw=0.7)
        ax[0].set_ylabel('PLMl (µV)')
        ax[0].set_title('Left Leg EMG (PLMl)')
        ax[0].grid(True, alpha=0.3)
        # PLMr
        ax[1].plot(times[start:end], data[1, start:end], color='r', lw=0.7)
        ax[1].set_ylabel('PLMr (µV)')
        ax[1].set_title('Right Leg EMG (PLMr)')
        ax[1].set_xlabel('Time (seconds)')
        ax[1].grid(True, alpha=0.3)
        # Clinical annotation lines (optional)
        for a in ax:
            a.axhline(0, color='k', lw=0.5, linestyle='--', alpha=0.5)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        # Save each window as a PNG
        out_png = OUTPUT_DIR / f"{edf_path.stem}_PLM_window{w+1}.png"
        plt.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"  Saved: {out_png}")
    print(f"  Visualization complete for {edf_path.name}")
except Exception as e:
    print(f"  Error processing {edf_path.name}: {e}")

print("\nAll visualizations saved in:", OUTPUT_DIR)
print("You can review the PNG files for clinical time series plots of PLMl and PLMr.") 