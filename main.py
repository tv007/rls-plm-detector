#!/usr/bin/env python3
"""
Main command-line interface for WASM 2019 leg movement detection software.

Usage:
    python main.py --input data/patient_data.edf --output reports/
    python main.py --batch data/ --output reports/
    python main.py --config config.yaml --input data/patient_data.edf
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from src.main_analysis import WASMAnalyzer, analyze_sleep_study, analyze_batch
from src.utils.data_structures import AnalysisConfig


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('wasm_analysis.log')
        ]
    )


def load_config(config_path: Optional[str] = None) -> AnalysisConfig:
    """Load analysis configuration from file or use defaults."""
    if config_path and Path(config_path).exists():
        import yaml
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return AnalysisConfig(**config_data)
    else:
        return AnalysisConfig()


def analyze_single_file(args):
    """Analyze a single sleep study file."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup output directory
    output_dir = Path(args.output) if args.output else Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract patient ID from filename if not provided
    patient_id = args.patient_id or Path(args.input).stem
    
    logger.info(f"Starting analysis of: {args.input}")
    logger.info(f"Patient ID: {patient_id}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Perform analysis
        result = analyze_sleep_study(
            file_path=args.input,
            annotations_path=args.annotations,
            patient_id=patient_id,
            recording_date=args.recording_date,
            config=config
        )
        
        # Save results
        analyzer = WASMAnalyzer(config)
        analyzer._save_result(result, str(output_dir), patient_id)
        
        # Print summary
        print("\n" + "="*50)
        print("ANALYSIS SUMMARY")
        print("="*50)
        print(f"Patient ID: {result.patient_id}")
        print(f"PLMS Diagnosis: {result.plms_diagnosis.upper()}")
        print(f"PLMS Index: {result.metrics.plms_index:.2f} per hour")
        print(f"Total CLMs: {result.metrics.total_clms}")
        print(f"Total PLMs: {result.metrics.total_plms}")
        print(f"Periodicity Index: {result.metrics.periodicity_index:.2f}")
        print(f"Clinical Significance: {result.clinical_significance}")
        print("\nRecommendations:")
        for rec in result.recommendations:
            print(f"  - {rec}")
        print("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return 1


def analyze_batch_files(args):
    """Analyze multiple sleep study files in batch."""
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_config(args.config)
    
    # Find all EDF files in input directory
    input_dir = Path(args.input)
    if not input_dir.exists():
        logger.error(f"Input directory does not exist: {input_dir}")
        return 1
    
    edf_files = list(input_dir.glob("*.edf")) + list(input_dir.glob("*.bdf")) + list(input_dir.glob("*.gdf"))
    
    if not edf_files:
        logger.error(f"No EDF/BDF/GDF files found in: {input_dir}")
        return 1
    
    logger.info(f"Found {len(edf_files)} files to analyze")
    
    # Setup output directory
    output_dir = Path(args.output) if args.output else Path("reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Perform batch analysis
        analyzer = WASMAnalyzer(config)
        results = analyzer.analyze_batch(
            file_paths=[str(f) for f in edf_files],
            output_dir=str(output_dir)
        )
        
        # Print batch summary
        print("\n" + "="*50)
        print("BATCH ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total files processed: {len(results)}/{len(edf_files)}")
        
        # Count diagnoses
        diagnoses = {}
        for result in results:
            diagnosis = result.plms_diagnosis
            diagnoses[diagnosis] = diagnoses.get(diagnosis, 0) + 1
        
        print("\nDiagnosis distribution:")
        for diagnosis, count in diagnoses.items():
            print(f"  {diagnosis.upper()}: {count}")
        
        print(f"\nResults saved to: {output_dir}")
        print("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {e}")
        return 1


def validate_data(args):
    """Validate sleep study data without performing full analysis."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.utils.data_loader import load_sleep_study
        
        logger.info(f"Validating data file: {args.input}")
        
        # Load data
        study_data = load_sleep_study(args.input, args.annotations)
        
        # Print validation results
        print("\n" + "="*50)
        print("DATA VALIDATION RESULTS")
        print("="*50)
        print(f"File: {args.input}")
        print(f"Duration: {study_data['duration']:.2f} seconds")
        print(f"Sampling frequency: {study_data['sampling_frequency']} Hz")
        print(f"Channels: {len(study_data['signals'])}")
        
        print("\nSignal channels:")
        for ch_name, signal in study_data['signals'].items():
            print(f"  - {ch_name}: {len(signal.data)} samples")
        
        if 'quality_metrics' in study_data:
            print("\nSignal quality:")
            for ch_name, quality in study_data['quality_metrics'].items():
                print(f"  - {ch_name}: {quality['overall_quality']:.2f}")
        
        if 'respiratory_events' in study_data:
            print(f"\nRespiratory events: {len(study_data['respiratory_events'])}")
        
        if 'arousal_events' in study_data:
            print(f"Arousal events: {len(study_data['arousal_events'])}")
        
        if 'sleep_stages' in study_data:
            print(f"Sleep stage segments: {len(study_data['sleep_stages'])}")
        
        print("="*50)
        
        return 0
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return 1


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="WASM 2019 Leg Movement Detection Software",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --input data/patient.edf --output reports/
  python main.py --batch data/ --output reports/
  python main.py --validate data/patient.edf
  python main.py --config config.yaml --input data/patient.edf
        """
    )
    
    # Global options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--config', type=str,
                       help='Path to configuration file (YAML)')
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Analyze single file
    parser_analyze = subparsers.add_parser('analyze', help='Analyze single sleep study file')
    parser_analyze.add_argument('--input', '-i', required=True,
                               help='Input sleep study file (EDF/BDF/GDF)')
    parser_analyze.add_argument('--output', '-o', default='reports',
                               help='Output directory for results')
    parser_analyze.add_argument('--annotations', type=str,
                               help='Path to annotations file (CSV/JSON)')
    parser_analyze.add_argument('--patient-id', type=str,
                               help='Patient identifier')
    parser_analyze.add_argument('--recording-date', type=str,
                               help='Recording date')
    
    # Batch analysis
    parser_batch = subparsers.add_parser('batch', help='Analyze multiple files in batch')
    parser_batch.add_argument('--input', '-i', required=True,
                             help='Input directory containing sleep study files')
    parser_batch.add_argument('--output', '-o', default='reports',
                             help='Output directory for results')
    
    # Data validation
    parser_validate = subparsers.add_parser('validate', help='Validate sleep study data')
    parser_validate.add_argument('--input', '-i', required=True,
                                help='Input sleep study file to validate')
    parser_validate.add_argument('--annotations', type=str,
                                help='Path to annotations file')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Execute command
    if args.command == 'analyze':
        return analyze_single_file(args)
    elif args.command == 'batch':
        return analyze_batch_files(args)
    elif args.command == 'validate':
        return validate_data(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 