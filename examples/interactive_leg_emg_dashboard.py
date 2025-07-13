#!/usr/bin/env python3
"""
Interactive GUI dashboard for visualizing PLMl and PLMr EMG signals from an EDF file.
Uses Streamlit for a clinical, user-friendly interface.
"""

import streamlit as st
import mne
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="Leg Movement EMG Dashboard", layout="wide")
st.title("Leg Movement EMG Clinical Dashboard")

DATA_DIR = Path("data")

# List available EDF files
df_files = sorted(DATA_DIR.glob("*.edf"))
file_names = [f.name for f in df_files]

if not file_names:
    st.error("No EDF files found in the data directory.")
    st.stop()

# File selection
selected_file = st.selectbox("Select an EDF file to visualize:", file_names)
edf_path = DATA_DIR / selected_file

# Load EDF and find PLMl/PLMr channels
@st.cache_data(show_spinner=True)
def load_leg_emg_channels(edf_path):
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    ch_names = [ch.lower() for ch in raw.ch_names]
    plml_idx = next((i for i, ch in enumerate(ch_names) if ch == 'plml'), None)
    plmr_idx = next((i for i, ch in enumerate(ch_names) if ch == 'plmr'), None)
    if plml_idx is None or plmr_idx is None:
        return None, None, None, None
    sfreq = raw.info['sfreq']
    n_samples = raw.n_times
    duration_sec = n_samples / sfreq
    data, times = raw[[plml_idx, plmr_idx], :]
    return data, times, sfreq, duration_sec

with st.spinner(f"Loading {selected_file}..."):
    data, times, sfreq, duration_sec = load_leg_emg_channels(edf_path)

if data is None:
    st.error("PLMl or PLMr channel not found in the selected EDF file.")
    st.stop()

# Time window selection
st.sidebar.header("Visualization Controls")
window_sec = st.sidebar.slider("Window size (seconds)", min_value=10, max_value=600, value=60, step=10)
start_time = st.sidebar.slider(
    "Start time (seconds)",
    min_value=0,
    max_value=int(duration_sec - window_sec),
    value=0,
    step=1
)
end_time = start_time + window_sec

# Downsample for performance if needed
DOWNSAMPLE_FACTOR = int(max(1, sfreq // 100))
if DOWNSAMPLE_FACTOR > 1:
    data = data[:, ::DOWNSAMPLE_FACTOR]
    times = times[::DOWNSAMPLE_FACTOR]
    sfreq = sfreq / DOWNSAMPLE_FACTOR

# Find indices for the selected window
start_idx = int(start_time * sfreq)
end_idx = int(end_time * sfreq)

# Plotting
fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
fig.suptitle(f"{selected_file} | PLMl & PLMr | {start_time}-{end_time} sec", fontsize=16)

ax[0].plot(times[start_idx:end_idx], data[0, start_idx:end_idx], color='b', lw=0.7)
ax[0].set_ylabel('PLMl (µV)')
ax[0].set_title('Left Leg EMG (PLMl)')
ax[0].grid(True, alpha=0.3)

ax[1].plot(times[start_idx:end_idx], data[1, start_idx:end_idx], color='r', lw=0.7)
ax[1].set_ylabel('PLMr (µV)')
ax[1].set_title('Right Leg EMG (PLMr)')
ax[1].set_xlabel('Time (seconds)')
ax[1].grid(True, alpha=0.3)

for a in ax:
    a.axhline(0, color='k', lw=0.5, linestyle='--', alpha=0.5)

plt.tight_layout(rect=[0, 0, 1, 0.96])
st.pyplot(fig)

st.info("Use the controls in the sidebar to change the time window and zoom in on different parts of the signal. This dashboard is optimized for clinical review of leg movement EMG signals.") 