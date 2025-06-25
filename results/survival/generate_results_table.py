import pandas as pd
import numpy as np
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def calculate_95_ci(c_indices):
    """Calculate 95% confidence interval for C-indices"""
    # Parse string representation of list if needed
    if isinstance(c_indices, str):
        c_indices = json.loads(c_indices)
    
    c_indices = np.array(c_indices)
    n = len(c_indices)
    
    if n < 2:
        return (c_indices[0], c_indices[0]) if n == 1 else (np.nan, np.nan)
    
    # Calculate confidence interval using t-distribution
    mean = np.mean(c_indices)
    std_err = stats.sem(c_indices)
    ci = stats.t.interval(0.95, n-1, loc=mean, scale=std_err)
    
    return ci

def format_ci(mean, ci_lower, ci_upper):
    """Format confidence interval string"""
    return f"{mean:.3f} ({ci_lower:.3f}-{ci_upper:.3f})"

def create_comprehensive_table():
    """Create comprehensive results table with 95% CI"""
    
    # Load the results
    df = pd.read_csv('survival_analysis_summary_v3.csv')
    
    # Calculate 95% CI for each row
    ci_results = []
    for _, row in df.iterrows():
        ci_lower, ci_upper = calculate_95_ci(row['c_indices'])
        ci_results.append({
            'project_id': row['project_id'],
            'merged_project': row['merged_project'],
            'modality': row['modality'],
            'model': row['model'],
            'mean_c_index': row['mean_c_index'],
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'formatted_result': format_ci(row['mean_c_index'], ci_lower, ci_upper)
        })
    
    results_df = pd.DataFrame(ci_results)
    
    # Create pivot tables for each model
    models = ['cox', 'rsf', 'deepsurv']
    
    # 1. Summary by Model and Modality (across all projects)
    print("="*80)
    print("SURVIVAL ANALYSIS RESULTS - MODEL PERFORMANCE BY MODALITY")
    print("="*80)
    print("\nC-index with 95% Confidence Intervals\n")
    
    modality_order = ['clinical', 'pathology', 'radiology', 'molecular', 'wsi', 
                      'concat', 'mean_pool', 'kronecker']
    
    for model in models:
        print(f"\n{model.upper()} Model:")
        print("-" * 60)
        
        model_data = results_df[results_df['model'] == model]
        summary = model_data.groupby('modality').agg({
            'mean_c_index': ['mean', 'count'],
            'formatted_result': lambda x: f"{x.iloc[0]}" if len(x) > 0 else "N/A"
        })
        
        # Calculate overall mean and CI across all projects for each modality
        for modality in modality_order:
            mod_data = model_data[model_data['modality'] == modality]
            if len(mod_data) > 0:
                all_c_indices = []
                for _, row in df[(df['model'] == model) & (df['modality'] == modality)].iterrows():
                    all_c_indices.extend(json.loads(row['c_indices']))
                
                if all_c_indices:
                    overall_mean = np.mean(all_c_indices)
                    overall_ci = calculate_95_ci(all_c_indices)
                    n_projects = len(mod_data)
                    print(f"  {modality:12s}: {format_ci(overall_mean, overall_ci[0], overall_ci[1]):20s} (n={n_projects} projects)")
    
    # 2. Create detailed table by cancer type
    print("\n\n" + "="*80)
    print("DETAILED RESULTS BY CANCER TYPE")
    print("="*80)
    
    # Group by merged project
    projects = sorted(results_df['merged_project'].unique())
    
    for project in projects:
        print(f"\n\n{project}")
        print("-" * 60)
        
        project_data = results_df[results_df['merged_project'] == project]
        
        # Create a table for this project
        table_data = []
        for modality in modality_order:
            if modality not in project_data['modality'].values:
                continue
                
            row = {'Modality': modality}
            for model in models:
                cell_data = project_data[(project_data['modality'] == modality) & 
                                       (project_data['model'] == model)]
                if len(cell_data) > 0:
                    row[model.upper()] = cell_data.iloc[0]['formatted_result']
                else:
                    row[model.upper()] = "N/A"
            table_data.append(row)
        
        if table_data:
            table_df = pd.DataFrame(table_data)
            print(table_df.to_string(index=False))
    
    # 3. Best performing configurations
    print("\n\n" + "="*80)
    print("TOP 10 BEST PERFORMING CONFIGURATIONS")
    print("="*80)
    print("\nRank | Cancer Type          | Modality    | Model    | C-index (95% CI)")
    print("-" * 80)
    
    top_configs = results_df.nlargest(10, 'mean_c_index')
    for i, row in enumerate(top_configs.iterrows(), 1):
        _, data = row
        print(f"{i:4d} | {data['merged_project']:20s} | {data['modality']:11s} | "
              f"{data['model']:8s} | {data['formatted_result']}")
    
    # 4. Summary statistics
    print("\n\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Overall performance by model
    print("\nOverall Model Performance (across all modalities and projects):")
    for model in models:
        model_data = results_df[results_df['model'] == model]
        all_c_indices = []
        for _, row in df[df['model'] == model].iterrows():
            all_c_indices.extend(json.loads(row['c_indices']))
        
        if all_c_indices:
            overall_mean = np.mean(all_c_indices)
            overall_ci = calculate_95_ci(all_c_indices)
            n_configs = len(model_data)
            print(f"  {model:8s}: {format_ci(overall_mean, overall_ci[0], overall_ci[1]):20s} "
                  f"(n={n_configs} configurations)")
    
    # Overall performance by modality
    print("\nOverall Modality Performance (across all models and projects):")
    for modality in modality_order:
        mod_data = results_df[results_df['modality'] == modality]
        if len(mod_data) > 0:
            all_c_indices = []
            for _, row in df[df['modality'] == modality].iterrows():
                all_c_indices.extend(json.loads(row['c_indices']))
            
            if all_c_indices:
                overall_mean = np.mean(all_c_indices)
                overall_ci = calculate_95_ci(all_c_indices)
                n_configs = len(mod_data)
                print(f"  {modality:12s}: {format_ci(overall_mean, overall_ci[0], overall_ci[1]):20s} "
                      f"(n={n_configs} configurations)")
    
    # Save detailed results to Excel
    print("\n\nSaving detailed results to 'survival_analysis_results_table.xlsx'...")
    
    with pd.ExcelWriter('survival_analysis_results_table.xlsx', engine='openpyxl') as writer:
        # Sheet 1: Full results with CI
        results_df.to_excel(writer, sheet_name='Full Results', index=False)
        
        # Sheet 2: Pivot by cancer type
        for project in projects[:10]:  # Limit to first 10 projects for Excel
            project_data = results_df[results_df['merged_project'] == project]
            pivot = project_data.pivot(index='modality', columns='model', values='formatted_result')
            pivot.to_excel(writer, sheet_name=f'{project[:20]}')
        
        # Sheet 3: Summary statistics
        summary_stats = []
        
        # By model
        for model in models:
            model_data = results_df[results_df['model'] == model]
            summary_stats.append({
                'Category': 'Model',
                'Name': model,
                'Mean C-index': model_data['mean_c_index'].mean(),
                'Min C-index': model_data['mean_c_index'].min(),
                'Max C-index': model_data['mean_c_index'].max(),
                'N Configurations': len(model_data)
            })
        
        # By modality
        for modality in modality_order:
            mod_data = results_df[results_df['modality'] == modality]
            if len(mod_data) > 0:
                summary_stats.append({
                    'Category': 'Modality',
                    'Name': modality,
                    'Mean C-index': mod_data['mean_c_index'].mean(),
                    'Min C-index': mod_data['mean_c_index'].min(),
                    'Max C-index': mod_data['mean_c_index'].max(),
                    'N Configurations': len(mod_data)
                })
        
        summary_df = pd.DataFrame(summary_stats)
        summary_df.to_excel(writer, sheet_name='Summary Statistics', index=False)
    
    # Save summary table as CSV
    results_df.to_csv('survival_analysis_results_with_ci.csv', index=False)
    
    print("\nResults saved to:")
    print("  - survival_analysis_results_with_ci.csv")
    print("  - survival_analysis_results_table.xlsx")
    
    # Create LaTeX table for top results
    print("\n\nLaTeX Table (Top 10 Results):")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Top 10 Survival Analysis Configurations}")
    print("\\begin{tabular}{llllc}")
    print("\\hline")
    print("Cancer Type & Modality & Model & C-index (95\\% CI) \\\\")
    print("\\hline")
    
    for _, row in top_configs.head(10).iterrows():
        # Escape underscores for LaTeX
        project = row['merged_project'].replace('_', '\\_')
        print(f"{project} & {row['modality']} & {row['model']} & {row['formatted_result']} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")

if __name__ == "__main__":
    create_comprehensive_table()