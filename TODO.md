# TODO List - WASM 2019 Leg Movement Detection Software

## Project Setup and Infrastructure
- [x] Create project structure and README
- [x] Create comprehensive TODO list
- [ ] Set up virtual environment
- [ ] Create requirements.txt with all dependencies
- [ ] Set up logging configuration
- [ ] Create configuration files for parameters
- [ ] Set up testing framework (pytest)
- [ ] Create data validation utilities

## Core Data Structures and Models
- [ ] Define EMG data structure class
- [ ] Define EEG data structure class
- [ ] Define respiratory events data structure
- [ ] Define leg movement event class
- [ ] Define PLM sequence class
- [ ] Define sleep stage enumeration
- [ ] Define diagnostic result class
- [ ] Create data loading utilities for EDF files

## STEP 1: Preprocessing Module
- [ ] Implement bandpass filter (10-100 Hz) for EMG
- [ ] Implement signal rectification
- [ ] Implement moving average smoothing (300ms window)
- [ ] Create preprocessing pipeline function
- [ ] Add signal quality assessment
- [ ] Add preprocessing visualization utilities
- [ ] Write unit tests for preprocessing functions

## STEP 2: Baseline Estimation Module
- [ ] Implement initial baseline calculation (first 30 seconds)
- [ ] Implement dynamic baseline estimation with 15s sliding windows
- [ ] Add baseline validation checks (>5µV warning, >16µV caution)
- [ ] Implement baseline smoothing and interpolation
- [ ] Add baseline visualization
- [ ] Write unit tests for baseline functions

## STEP 3: Leg Movement Detection Module
- [ ] Implement amplitude threshold detection (baseline + 8µV)
- [ ] Implement duration validation (≥0.5s)
- [ ] Implement median amplitude check (≥baseline + 2µV)
- [ ] Create LM event detection pipeline
- [ ] Add LM event data structure
- [ ] Add LM detection visualization
- [ ] Write unit tests for LM detection

## STEP 4: Candidate Leg Movement Filtering
- [ ] Implement duration filtering (0.5s ≤ duration ≤ 10s)
- [ ] Create CLM data structure
- [ ] Add CLM validation utilities
- [ ] Write unit tests for CLM filtering

## STEP 5: Bilateral CLM Combination
- [ ] Implement temporal alignment check (≤0.5s difference)
- [ ] Implement combined duration validation (≤15s total)
- [ ] Implement bilateral CLM merging logic
- [ ] Create bilateral CLM data structure
- [ ] Add bilateral CLM visualization
- [ ] Write unit tests for bilateral combination

## STEP 6: Periodic Leg Movement Sequence Detection
- [ ] Implement IMI calculation (10s ≤ IMI ≤ 90s)
- [ ] Implement PLM sequence detection (≥4 movements)
- [ ] Create PLM sequence data structure
- [ ] Add PLM sequence validation
- [ ] Add PLM sequence visualization
- [ ] Write unit tests for PLM detection

## STEP 7: Respiratory Association Module
- [ ] Implement respiratory event data loading
- [ ] Implement temporal association logic (-2s to +10.25s)
- [ ] Create CLMr marking system
- [ ] Add respiratory association visualization
- [ ] Write unit tests for respiratory association

## STEP 8: Arousal Association Module
- [ ] Implement arousal detection/loading
- [ ] Implement overlap detection logic
- [ ] Implement temporal proximity check (≤0.5s)
- [ ] Create PLMSa marking system
- [ ] Add arousal association visualization
- [ ] Write unit tests for arousal association

## STEP 9: Sleep Stage Segmentation
- [ ] Implement sleep stage data loading
- [ ] Implement stage-based CLM categorization
- [ ] Create stage-specific CLM collections
- [ ] Add sleep stage visualization
- [ ] Write unit tests for sleep staging

## STEP 10: Metrics Calculation Module
- [ ] Implement PLMS index calculation (per hour)
- [ ] Implement PLMW index calculation (wake movements per hour)
- [ ] Implement periodicity index calculation
- [ ] Implement mean duration calculation
- [ ] Implement IMI distribution analysis
- [ ] Create comprehensive metrics data structure
- [ ] Add metrics validation
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
- [ ] Create main analysis pipeline
- [ ] Implement command-line interface
- [ ] Add configuration file support
- [ ] Implement error handling and recovery
- [ ] Add progress tracking and logging
- [ ] Create batch processing capabilities
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
- **Completed**: 2/50 tasks (4%)
- **In Progress**: 0/50 tasks
- **Remaining**: 48/50 tasks (96%)

## Notes
- Priority should be given to core detection algorithms (Steps 1-6)
- Testing should be implemented alongside development
- Documentation should be maintained throughout development
- Performance optimization should be considered from the start 