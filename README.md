# WASM 2019 Leg Movement Detection Software


## Overview
This software implements automatic detection of leg movements according to the World Association of Sleep Medicine (WASM) 2019 guidelines and provides diagnostic output for sleep disorders.

## Features
- EMG signal preprocessing and filtering
- Automatic leg movement detection
- Periodic leg movement (PLM) sequence analysis
- Respiratory and arousal association
- Sleep stage segmentation
- Comprehensive diagnostic reporting
- Real-time analysis capabilities

## Project Structure
```
software/
├── src/
│   ├── preprocessing/
│   ├── detection/
│   ├── analysis/
│   ├── reporting/
│   └── utils/
├── tests/
├── data/
├── docs/
├── requirements.txt
└── main.py
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py --input data/patient_data.edf --output reports/
```

## License
MIT License 