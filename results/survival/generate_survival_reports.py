#!/usr/bin/env python3
"""
Generate comprehensive survival analysis reports and tables.
This script creates detailed performance reports, statistical analyses, and LaTeX tables.
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
from glob import glob
from typing import Dict, List, Tuple
from datetime import datetime
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_cv_results(cv_results_dir: str) -> pd.DataFrame:
    """Load all CV results and compile into a DataFrame."""
    
    cv_files = glob(os.path.join(cv_results_dir, '*.pkl'))
    results = []
    
    for cv_file in cv_files:
        filename = os.path.basename(cv_file)
        # Parse filename: project_modality_model_cv_results.pkl
        parts = filename.replace('_cv_results.pkl', '').split('_')
        
        # Handle merged projects
        if parts[0].startswith('TCGA-'):
            tcga_parts = []
            idx = 0
            while idx < len(parts) and (parts[idx].startswith('TCGA-') or 
                                       (idx > 0 and parts[idx-1].startswith('TCGA-'))):
                tcga_parts.append(parts[idx])
                idx += 1
            
            project = '_'.join(tcga_parts)
            modality = parts[idx] if idx < len(parts) else parts[-2]
            model_type = parts[-1]
        else:
            project = parts[0]
            modality = parts[1]
            model_type = parts[2]
        
        # Load CV results
        with open(cv_file, 'rb') as f:
            cv_data = pickle.load(f)
        
        result = {
            'project': project,
            'modality': modality,
            'model': model_type,
            'mean_c_index': cv_data.get('mean_c_index', np.nan),
            'std_c_index': cv_data.get('std_c_index', np.nan),
            'c_indices': cv_data.get('c_indices', []),
            'n_folds': len(cv_data.get('c_indices', [])),
            'train_c_indices': cv_data.get('train_c_indices', [])
        }
        
        # Calculate additional metrics
        if result['c_indices']:
            result['min_c_index'] = np.min(result['c_indices'])
            result['max_c_index'] = np.max(result['c_indices'])
            result['median_c_index'] = np.median(result['c_indices'])
            
            # Calculate 95% CI
            if len(result['c_indices']) > 1:
                result['ci_lower'] = np.percentile(result['c_indices'], 2.5)
                result['ci_upper'] = np.percentile(result['c_indices'], 97.5)
            else:
                result['ci_lower'] = result['mean_c_index']
                result['ci_upper'] = result['mean_c_index']
        
        results.append(result)
    
    return pd.DataFrame(results)


def perform_statistical_tests(results_df: pd.DataFrame) -> Dict:
    """Perform statistical tests comparing models and modalities."""
    
    tests = {}
    
    # 1. Compare models across all data
    model_comparison = {}
    models = results_df['model'].unique()
    
    for i, model1 in enumerate(models):
        for model2 in models[i+1:]:
            # Get c-indices for each model
            c1 = results_df[results_df['model'] == model1]['mean_c_index'].values
            c2 = results_df[results_df['model'] == model2]['mean_c_index'].values
            
            # Paired t-test (same projects and modalities)
            common_idx = results_df[results_df['model'] == model1][['project', 'modality']].merge(
                results_df[results_df['model'] == model2][['project', 'modality']], 
                on=['project', 'modality']
            ).index
            
            if len(common_idx) > 0:
                c1_paired = results_df[(results_df['model'] == model1) & 
                                     (results_df.index.isin(common_idx))]['mean_c_index'].values
                c2_paired = results_df[(results_df['model'] == model2) & 
                                     (results_df.index.isin(common_idx))]['mean_c_index'].values
                
                if len(c1_paired) > 1:
                    t_stat, p_value = stats.ttest_rel(c1_paired, c2_paired)
                    model_comparison[f'{model1}_vs_{model2}'] = {
                        'n_pairs': len(c1_paired),
                        'mean_diff': np.mean(c1_paired - c2_paired),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05
                    }
    
    tests['model_comparison'] = model_comparison
    
    # 2. Compare modalities
    modality_comparison = {}
    modalities = results_df['modality'].unique()
    
    for i, mod1 in enumerate(modalities):
        for mod2 in modalities[i+1:]:
            c1 = results_df[results_df['modality'] == mod1]['mean_c_index'].values
            c2 = results_df[results_df['modality'] == mod2]['mean_c_index'].values
            
            if len(c1) > 1 and len(c2) > 1:
                t_stat, p_value = stats.ttest_ind(c1, c2)
                modality_comparison[f'{mod1}_vs_{mod2}'] = {
                    'n1': len(c1),
                    'n2': len(c2),
                    'mean1': np.mean(c1),
                    'mean2': np.mean(c2),
                    'mean_diff': np.mean(c1) - np.mean(c2),
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    tests['modality_comparison'] = modality_comparison
    
    # 3. ANOVA across cancer types
    cancer_anova = {}
    for modality in modalities:
        mod_data = results_df[results_df['modality'] == modality]
        projects = mod_data['project'].unique()
        
        if len(projects) > 2:
            groups = [mod_data[mod_data['project'] == p]['mean_c_index'].values 
                     for p in projects if len(mod_data[mod_data['project'] == p]) > 0]
            groups = [g for g in groups if len(g) > 0]
            
            if len(groups) > 2:
                f_stat, p_value = stats.f_oneway(*groups)
                cancer_anova[modality] = {
                    'n_groups': len(groups),
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
    
    tests['cancer_type_anova'] = cancer_anova
    
    return tests


def generate_detailed_report(results_df: pd.DataFrame, output_path: str):
    """Generate a detailed markdown report."""
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []
    
    # Header
    report_lines.append("# Survival Analysis Detailed Report")
    report_lines.append(f"\nGenerated on: {timestamp}")
    report_lines.append(f"\nTotal experiments: {len(results_df)}")
    
    # Overall summary
    report_lines.append("\n## Overall Performance Summary")
    report_lines.append(f"\n- **Mean C-index across all experiments**: {results_df['mean_c_index'].mean():.3f} ± {results_df['mean_c_index'].std():.3f}")
    report_lines.append(f"- **Best performing configuration**: {results_df.loc[results_df['mean_c_index'].idxmax()]['modality']} - "
                       f"{results_df.loc[results_df['mean_c_index'].idxmax()]['model']} - "
                       f"{results_df.loc[results_df['mean_c_index'].idxmax()]['project']} "
                       f"(C-index: {results_df['mean_c_index'].max():.3f})")
    
    # Model comparison
    report_lines.append("\n## Model Performance Comparison")
    model_summary = results_df.groupby('model').agg({
        'mean_c_index': ['mean', 'std', 'min', 'max', 'count']
    }).round(3)
    
    report_lines.append("\n| Model | Mean C-index | Std | Min | Max | Count |")
    report_lines.append("|-------|--------------|-----|-----|-----|-------|")
    
    for model in model_summary.index:
        stats = model_summary.loc[model, 'mean_c_index']
        report_lines.append(f"| {model.upper()} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                          f"{stats['min']:.3f} | {stats['max']:.3f} | {int(stats['count'])} |")
    
    # Modality comparison
    report_lines.append("\n## Modality Performance Comparison")
    modality_summary = results_df.groupby('modality').agg({
        'mean_c_index': ['mean', 'std', 'min', 'max', 'count']
    }).round(3)
    
    report_lines.append("\n| Modality | Mean C-index | Std | Min | Max | Count |")
    report_lines.append("|----------|--------------|-----|-----|-----|-------|")
    
    for modality in modality_summary.index:
        stats = modality_summary.loc[modality, 'mean_c_index']
        report_lines.append(f"| {modality} | {stats['mean']:.3f} | {stats['std']:.3f} | "
                          f"{stats['min']:.3f} | {stats['max']:.3f} | {int(stats['count'])} |")
    
    # Statistical tests
    stat_tests = perform_statistical_tests(results_df)
    
    report_lines.append("\n## Statistical Comparisons")
    
    # Model comparisons
    report_lines.append("\n### Model Comparisons (Paired t-tests)")
    report_lines.append("\n| Comparison | Mean Difference | t-statistic | p-value | Significant |")
    report_lines.append("|------------|-----------------|-------------|---------|-------------|")
    
    for comp, stats in stat_tests['model_comparison'].items():
        report_lines.append(f"| {comp} | {stats['mean_diff']:.3f} | {stats['t_statistic']:.3f} | "
                          f"{stats['p_value']:.4f} | {'Yes' if stats['significant'] else 'No'} |")
    
    # Top performers by cancer type
    report_lines.append("\n## Top Performers by Cancer Type")
    
    cancer_types = results_df['project'].unique()
    for cancer in sorted(cancer_types):
        cancer_data = results_df[results_df['project'] == cancer]
        if len(cancer_data) > 0:
            best = cancer_data.loc[cancer_data['mean_c_index'].idxmax()]
            report_lines.append(f"\n### {cancer}")
            report_lines.append(f"- **Best configuration**: {best['modality']} - {best['model']}")
            report_lines.append(f"- **C-index**: {best['mean_c_index']:.3f} ± {best['std_c_index']:.3f}")
            if 'ci_lower' in best:
                report_lines.append(f"- **95% CI**: [{best['ci_lower']:.3f}, {best['ci_upper']:.3f}]")
    
    # Save report
    report_path = os.path.join(output_path, 'survival_analysis_detailed_report.md')
    with open(report_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Detailed report saved to: {report_path}")


def generate_performance_csv(results_df: pd.DataFrame, output_path: str):
    """Generate comprehensive CSV with all performance metrics."""
    
    # Expand the dataframe with additional metrics
    expanded_results = results_df.copy()
    
    # Add additional columns
    expanded_results['ci_width'] = expanded_results['ci_upper'] - expanded_results['ci_lower']
    expanded_results['cv_coefficient'] = expanded_results['std_c_index'] / expanded_results['mean_c_index']
    
    # Sort by mean C-index
    expanded_results = expanded_results.sort_values('mean_c_index', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_path, 'survival_analysis_full_results.csv')
    expanded_results.to_csv(csv_path, index=False)
    
    print(f"Full results CSV saved to: {csv_path}")
    
    # Create a summary pivot table
    pivot_table = results_df.pivot_table(
        values='mean_c_index',
        index='project',
        columns=['modality', 'model'],
        aggfunc='first'
    ).round(3)
    
    pivot_path = os.path.join(output_path, 'survival_analysis_pivot_table.csv')
    pivot_table.to_csv(pivot_path)
    
    print(f"Pivot table saved to: {pivot_path}")


def generate_latex_performance_table(results_df: pd.DataFrame, output_path: str):
    """Generate publication-ready LaTeX table with confidence intervals."""
    
    latex_lines = []
    latex_lines.append(r'\begin{table}[htbp]')
    latex_lines.append(r'    \centering')
    latex_lines.append(r'    \caption{Survival analysis model performance comparison with 95\% confidence intervals}')
    latex_lines.append(r'    \label{tab:survival_performance_ci}')
    latex_lines.append(r'    \begin{tabular}{llcc}')
    latex_lines.append(r'        \toprule')
    latex_lines.append(r'        \textbf{Modality} & \textbf{Model} & \textbf{C-index} & \textbf{95\% CI} \\')
    latex_lines.append(r'        \midrule')
    
    # Group by modality
    modality_order = ['clinical', 'pathology', 'radiology', 'molecular', 'wsi', 
                     'concat', 'mean_pool', 'kronecker']
    model_order = ['cox', 'rsf', 'deepsurv']
    
    for modality in modality_order:
        if modality not in results_df['modality'].values:
            continue
            
        first_row = True
        mod_data = results_df[results_df['modality'] == modality]
        
        for model in model_order:
            model_data = mod_data[mod_data['model'] == model]
            
            if len(model_data) > 0:
                # Calculate aggregate statistics across all projects
                all_c_indices = []
                for _, row in model_data.iterrows():
                    if isinstance(row['c_indices'], list):
                        all_c_indices.extend(row['c_indices'])
                
                if all_c_indices:
                    mean_c = np.mean(all_c_indices)
                    ci_lower = np.percentile(all_c_indices, 2.5)
                    ci_upper = np.percentile(all_c_indices, 97.5)
                    
                    row_text = '        '
                    if first_row:
                        row_text += f'\\textbf{{{modality.capitalize()}}} & '
                    else:
                        row_text += ' & '
                    
                    row_text += f'{model.upper()} & '
                    row_text += f'{mean_c:.3f} & '
                    row_text += f'[{ci_lower:.3f}, {ci_upper:.3f}] \\\\'
                    
                    latex_lines.append(row_text)
                    first_row = False
        
        if not first_row:  # Only add midrule if we added rows
            latex_lines.append(r'        \midrule')
    
    # Remove last midrule
    if latex_lines[-1] == r'        \midrule':
        latex_lines.pop()
    
    latex_lines.append(r'        \bottomrule')
    latex_lines.append(r'    \end{tabular}')
    latex_lines.append(r'\end{table}')
    
    # Save table
    latex_path = os.path.join(output_path, 'survival_performance_ci_table.tex')
    with open(latex_path, 'w') as f:
        f.write('\n'.join(latex_lines))
    
    print(f"LaTeX performance table saved to: {latex_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate survival analysis reports')
    parser.add_argument('--output-path', type=str,
                       default='/mnt/f/Projects/HoneyBee/results/survival',
                       help='Path to output directory')
    parser.add_argument('--format', nargs='+', 
                       choices=['markdown', 'csv', 'latex', 'all'],
                       default=['all'],
                       help='Report formats to generate')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("GENERATING SURVIVAL ANALYSIS REPORTS")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    
    # Load CV results
    cv_results_dir = os.path.join(args.output_path, 'cv_results')
    results_df = load_cv_results(cv_results_dir)
    
    print(f"\nLoaded {len(results_df)} experiment results")
    
    # Generate reports based on format
    if 'all' in args.format or 'markdown' in args.format:
        generate_detailed_report(results_df, args.output_path)
    
    if 'all' in args.format or 'csv' in args.format:
        generate_performance_csv(results_df, args.output_path)
    
    if 'all' in args.format or 'latex' in args.format:
        generate_latex_performance_table(results_df, args.output_path)
    
    print("\n" + "=" * 80)
    print("REPORT GENERATION COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()