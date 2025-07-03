#!/usr/bin/env python3
"""
Generate survival analysis figures from pre-trained models.
This script loads saved models and creates all visualizations.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from survival_analysis import SurvivalAnalysis, DeepSurvModel


def load_saved_model(model_path: str) -> Dict:
    """Load a saved model from disk."""
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def generate_risk_curves_from_saved(analyzer: SurvivalAnalysis, modality: str, 
                                  model_type: str, project_id: str, 
                                  model_path: str):
    """Generate risk curves using a saved model."""
    
    print(f"\nGenerating risk curves for {modality} - {model_type} - {project_id}")
    
    # Load saved model
    model_data = load_saved_model(model_path)
    
    # Get embeddings
    if modality in analyzer.embeddings:
        embeddings = analyzer.embeddings[modality]
    else:
        embeddings = analyzer.multimodal_embeddings[modality]
    
    # Get project indices
    merged_project = model_data['project']
    _, project_indices = analyzer.merge_similar_projects(project_id)
    
    # Prepare survival data
    X, y, patient_subset = analyzer.prepare_survival_data(embeddings, project_indices)
    
    # Get model and scaler
    model = model_data['model']
    scaler = model_data['scaler']
    
    if model is None:
        print(f"  Skipping - no valid model")
        return
    
    # Create CV results dict for compatibility
    cv_results = {
        'project_name': merged_project,
        'mean_c_index': model_data['test_c_index'],
        'std_c_index': 0.0,  # Single model, no std
        'c_indices': [model_data['test_c_index']],
        'fold_models': [(model, scaler)]
    }
    
    # Use existing method to generate curves
    analyzer.generate_risk_curves(modality, model_type, project_id, cv_results)


def generate_all_risk_curves(data_path: str, output_path: str, 
                           regenerate_existing: bool = False):
    """Generate risk curves for all available models."""
    
    print("=" * 80)
    print("GENERATING SURVIVAL RISK CURVES")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    # Initialize analyzer
    analyzer = SurvivalAnalysis(data_path, output_path)
    analyzer.load_data()
    analyzer.create_multimodal_embeddings()
    
    # Find all saved models
    model_dir = os.path.join(output_path, 'models')
    best_models = glob(os.path.join(model_dir, '*_best.pkl'))
    
    print(f"\nFound {len(best_models)} best models")
    
    generated_count = 0
    skipped_count = 0
    
    for model_path in sorted(best_models):
        model_filename = os.path.basename(model_path)
        
        # Parse filename more carefully
        # Format: project_modality_model_best.pkl
        # But project can contain underscores (merged projects)
        
        # First, load the model to get metadata
        try:
            model_data = load_saved_model(model_path)
            project = model_data.get('project')
            modality = model_data.get('modality')
            model_type = model_data.get('model_type')
            
            if not all([project, modality, model_type]):
                # Fallback to filename parsing
                parts = model_filename.replace('_best.pkl', '').rsplit('_', 2)
                if len(parts) >= 3:
                    project = parts[0]
                    modality = parts[1]
                    model_type = parts[2]
                else:
                    print(f"\nError parsing filename: {model_filename}")
                    continue
        except Exception as e:
            print(f"\nError loading model {model_filename}: {str(e)}")
            continue
        
        # Get original project ID (first one in merged name)
        original_project = project.split('_')[0]
        
        # Check if figure already exists
        figure_dir = os.path.join(output_path, 'risk_curves', project)
        figure_path = os.path.join(figure_dir, f'{modality}_{model_type}_risk_stratification.pdf')
        
        if os.path.exists(figure_path) and not regenerate_existing:
            print(f"\nSkipping {modality} - {model_type} - {project} (figure exists)")
            skipped_count += 1
            continue
        
        print(f"\nGenerating {modality} - {model_type} - {project}")
        
        try:
            generate_risk_curves_from_saved(analyzer, modality, model_type, 
                                          original_project, model_path)
            generated_count += 1
        except Exception as e:
            print(f"  Error: {str(e)}")
    
    print("\n" + "=" * 80)
    print("FIGURE GENERATION COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now()}")
    print(f"Figures generated: {generated_count}")
    print(f"Figures skipped: {skipped_count}")


def generate_summary_visualizations(data_path: str, output_path: str):
    """Generate summary plots from all results."""
    
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY VISUALIZATIONS")
    print("=" * 80)
    
    # Find most recent training results
    result_files = glob(os.path.join(output_path, 'training_results_*.csv'))
    
    if not result_files:
        # Try the default summary file
        summary_file = os.path.join(output_path, 'survival_analysis_summary_v3.csv')
        if os.path.exists(summary_file):
            result_files = [summary_file]
        else:
            print("No training results found!")
            return
    
    # Use most recent results
    latest_results = sorted(result_files)[-1]
    print(f"Using results from: {os.path.basename(latest_results)}")
    
    results_df = pd.read_csv(latest_results)
    
    # Initialize analyzer for LaTeX table generation
    analyzer = SurvivalAnalysis(data_path, output_path)
    
    # Generate summary plots
    analyzer.generate_summary_plots(results_df)
    print("Summary plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(description='Generate survival analysis figures')
    parser.add_argument('--data-path', type=str, 
                       default='/mnt/f/Projects/HoneyBee/results/shared_data',
                       help='Path to shared data directory')
    parser.add_argument('--output-path', type=str,
                       default='/mnt/f/Projects/HoneyBee/results/survival',
                       help='Path to output directory')
    parser.add_argument('--regenerate', action='store_true',
                       help='Regenerate existing figures')
    parser.add_argument('--summary-only', action='store_true',
                       help='Only generate summary visualizations')
    parser.add_argument('--curves-only', action='store_true',
                       help='Only generate risk curves')
    
    args = parser.parse_args()
    
    if not args.summary_only:
        # Generate risk curves
        generate_all_risk_curves(
            data_path=args.data_path,
            output_path=args.output_path,
            regenerate_existing=args.regenerate
        )
    
    if not args.curves_only:
        # Generate summary visualizations
        generate_summary_visualizations(
            data_path=args.data_path,
            output_path=args.output_path
        )


if __name__ == "__main__":
    main()