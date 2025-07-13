# TODO List - WASM 2019 Leg Movement Detection Software

## Project Setup and Infrastructure
- [x] Create project structure and README
- [x] Create comprehensive TODO list
- [x] Set up virtual environment
- [x] Create requirements.txt with all dependencies
- [x] Set up logging configuration
- [x] Create configuration files for parameters
- [ ] Set up testing framework (pytest)
- [x] Create data validation utilities

## Core Data Structures and Models
- [x] Define EMG data structure class
- [x] Define EEG data structure class
- [x] Define respiratory events data structure
- [x] Define leg movement event class
- [x] Define PLM sequence class
- [x] Define sleep stage enumeration
- [x] Define diagnostic result class
- [x] Create data loading utilities for EDF files
- [x] **Update: Only PLMl and PLMr channels are used for leg movement detection. Channel selection logic corrected to use only these.**

## STEP 1: Preprocessing Module
- [x] Implement bandpass filter (10-100 Hz) for EMG
- [x] Implement signal rectification
- [x] Implement moving average smoothing (300ms window)
- [x] Create preprocessing pipeline function
- [x] Add signal quality assessment
- [ ] Add preprocessing visualization utilities
- [ ] Write unit tests for preprocessing functions
- [x] **Update: Channel selection for leg movement detection now strictly uses PLMl and PLMr channels.**

## STEP 2: Baseline Estimation Module
- [x] Implement initial baseline calculation (first 30 seconds)
- [x] Implement dynamic baseline estimation with 15s sliding windows
- [x] Add baseline validation checks (>5µV warning, >16µV caution)
- [ ] Implement baseline smoothing and interpolation
- [ ] Add baseline visualization
- [ ] Write unit tests for baseline functions

## STEP 3: Leg Movement Detection Module
- [x] Implement amplitude threshold detection (baseline + 8µV)
- [x] Implement duration validation (≥0.5s)
- [x] Implement median amplitude check (≥baseline + 2µV)
- [x] Create LM event detection pipeline
- [x] Add LM event data structure
- [ ] Add LM detection visualization
- [ ] Write unit tests for LM detection

## STEP 4: Candidate Leg Movement Filtering
- [x] Implement duration filtering (0.5s ≤ duration ≤ 10s)
- [x] Create CLM data structure
- [x] Add CLM validation utilities
- [ ] Write unit tests for CLM filtering

## STEP 5: Bilateral CLM Combination
- [x] Implement temporal alignment check (≤0.5s difference)
- [x] Implement combined duration validation (≤15s total)
- [x] Implement bilateral CLM merging logic
- [x] Create bilateral CLM data structure
- [ ] Add bilateral CLM visualization
- [ ] Write unit tests for bilateral combination

## STEP 6: Periodic Leg Movement Sequence Detection
- [x] Implement IMI calculation (10s ≤ IMI ≤ 90s)
- [x] Implement PLM sequence detection (≥4 movements)
- [x] Create PLM sequence data structure
- [x] Add PLM sequence validation
- [ ] Add PLM sequence visualization
- [ ] Write unit tests for PLM detection

## STEP 7: Respiratory Association Module
- [x] Implement respiratory event data loading
- [x] Implement temporal association logic (-2s to +10.25s)
- [x] Create CLMr marking system
- [ ] Add respiratory association visualization
- [ ] Write unit tests for respiratory association

## STEP 8: Arousal Association Module
- [x] Implement arousal detection/loading
- [x] Implement overlap detection logic
- [x] Implement temporal proximity check (≤0.5s)
- [x] Create PLMSa marking system
- [ ] Add arousal association visualization
- [ ] Write unit tests for arousal association

## STEP 9: Sleep Stage Segmentation
- [x] Implement sleep stage data loading
- [x] Implement stage-based CLM categorization
- [x] Create stage-specific CLM collections
- [ ] Add sleep stage visualization
- [ ] Write unit tests for sleep staging

## STEP 10: Metrics Calculation Module
- [x] Implement PLMS index calculation (per hour)
- [x] Implement PLMW index calculation (wake movements per hour)
- [x] Implement periodicity index calculation
- [x] Implement mean duration calculation
- [x] Implement IMI distribution analysis
- [x] Create comprehensive metrics data structure
- [x] Add metrics validation
- [ ] Write unit tests for metrics calculation

## STEP 11: Diagnostic Reporting Module
- [ ] Implement summary table generation
- [ ] Implement EMG signal plotting with PLM overlays
- [ ] Implement PDF report generation
- [ ] Implement JSON/CSV export functionality
- [ ] Create diagnostic criteria evaluation
- [ ] Add report customization options
- [ ] Write unit tests for reporting

## Integration and Main Application
- [x] Create main analysis pipeline
- [x] Implement command-line interface
- [x] Add configuration file support
- [x] Implement error handling and recovery
- [x] Add progress tracking and logging
- [x] Create batch processing capabilities
- [ ] Add real-time analysis mode
- [ ] Write integration tests

## Performance and Optimization
- [ ] Optimize signal processing algorithms
- [ ] Implement parallel processing for large datasets
- [ ] Add memory management for large files
- [ ] Optimize visualization rendering
- [ ] Add caching mechanisms

## Documentation and Examples
- [ ] Write comprehensive API documentation
- [ ] Create user manual
- [ ] Add code comments and docstrings
- [ ] Create example datasets
- [ ] Write tutorial notebooks
- [ ] Create deployment guide
- [x] **Add: Clinical-grade visualization of PLMl and PLMr EMG signals as time series, with efficient handling of large signals and dashboard features (see examples/visualize_leg_emg.py)**
- [x] **Add: Interactive GUI dashboard for PLMl and PLMr EMG visualization using Streamlit (see examples/interactive_leg_emg_dashboard.py)**

## Testing and Validation
- [ ] Create synthetic test datasets
- [ ] Implement validation against manual scoring
- [ ] Add performance benchmarking
- [ ] Create regression tests
- [ ] Add edge case testing
- [ ] Implement continuous integration

## Deployment and Distribution
- [ ] Create Docker container
- [ ] Set up PyPI package
- [ ] Create installation scripts
- [ ] Add system requirements check
- [ ] Create deployment documentation

## Future Enhancements
- [ ] Add machine learning-based detection
- [ ] Implement real-time streaming analysis
- [ ] Add cloud-based processing
- [ ] Create web interface
- [ ] Add mobile app support
- [ ] Implement multi-center validation

## Progress Tracking
- **Completed**: 35/50 tasks (70%)
- **In Progress**: 0/50 tasks
- **Remaining**: 15/50 tasks (30%)

## Notes
- Priority should be given to core detection algorithms (Steps 1-6)
- Testing should be implemented alongside development
- Documentation should be maintained throughout development
- Performance optimization should be considered from the start 