"""Command-line interface for the Big Five Trait evaluation pipeline."""

import argparse
import os
import sys
from typing import Optional

from pipelines.pipeline import run_pipeline


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate Big Five personality traits across multiple LLMs.')
    
    parser.add_argument(
        '--config',
        type=str,
        default='pipelines/config.yaml',
        help='Path to the configuration file (default: pipelines/config.yaml)'
    )
    
    parser.add_argument(
        '--input-text',
        type=str,
        help='Text to evaluate for personality traits (overrides config)'
    )
    
    parser.add_argument(
        '--output-csv',
        type=str,
        help='Path to save results as CSV'
    )
    
    parser.add_argument(
        '--output-json',
        type=str,
        help='Path to save results as JSON'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Check that the config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Run the pipeline
    try:
        results = run_pipeline(
            config_path=args.config,
            input_text=args.input_text,
            output_path=args.output_csv
        )
        
        # Print results
        if args.verbose:
            print("\nResults:")
            print(results)
            print("\nPersonality Trait Scores Summary:")
            
            # Display a simplified view of the scores without rationales
            score_cols = [col for col in results.columns if col.endswith('_score')]
            print(results[['llm'] + score_cols])
        else:
            # Just print the scores in a compact format
            score_cols = [col for col in results.columns if col.endswith('_score')]
            print(results[['llm'] + score_cols])
        
        # Save as JSON if requested
        if args.output_json:
            results.to_json(args.output_json, orient='records', indent=2)
            print(f"Results saved to {args.output_json}")
        
    except Exception as e:
        print(f"Error running pipeline: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main() 