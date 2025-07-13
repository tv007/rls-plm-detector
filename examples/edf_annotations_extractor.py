import os
import json
import mne

DATA_DIR = 'data'

filename = 'Gangwar_Kusum_(1).edf'
edf_path = os.path.join(DATA_DIR, filename)
patched_path = os.path.join(DATA_DIR, 'Gangwar_Kusum_(1)_patched.edf')
json_path = os.path.join(DATA_DIR, os.path.splitext(filename)[0] + '_annotations.json')

# Patch the header: replace '510p8' with '51088' (must be same length)
try:
    with open(edf_path, 'rb') as f:
        content = f.read()
    # Find and replace only the first occurrence in the header (first 512 bytes)
    header = content[:512]
    rest = content[512:]
    if b'510p8' in header:
        patched_header = header.replace(b'510p8', b'51088', 1)
        with open(patched_path, 'wb') as f:
            f.write(patched_header + rest)
        print(f"Patched file written to {patched_path}")
    else:
        print("Pattern '510p8' not found in header; no patch applied.")
except Exception as e:
    print(f"Failed to patch header: {e}")

# Try to read metadata from the patched file
try:
    print(f"Attempting to read metadata from patched file {patched_path}...")
    raw = mne.io.read_raw_edf(patched_path, preload=False, verbose=True)
    print("\n--- Metadata ---")
    print(f"Channel names: {raw.ch_names}")
    print(f"Sampling frequency: {raw.info['sfreq']}")
    print(f"Number of samples: {raw.n_times}")
    print(f"Measurement date: {raw.info['meas_date']}")
    print(f"Duration (seconds): {raw.times[-1] if hasattr(raw, 'times') else 'N/A'}")
    print(f"Info: {raw.info}")
    if hasattr(raw, 'annotations') and raw.annotations is not None:
        print(f"Annotations: {raw.annotations}")
    else:
        print("No annotations found.")
except Exception as e:
    print(f"Failed to read metadata from patched file: {e}") 