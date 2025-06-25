#!/usr/bin/env python3
"""
Generate markdown and LaTeX tables for survival analysis C-index results.
"""

import pandas as pd
import numpy as np

# Load the survival analysis summary
df = pd.read_csv('survival_analysis_summary_v3.csv')

# Get unique projects and modalities
projects = df['merged_project'].unique()
modalities = ['clinical', 'pathology', 'radiology', 'molecular', 'wsi', 'concat', 'mean_pool', 'kronecker']

# For this analysis, we'll focus on Cox model results
model_type = 'cox'

# Create a pivot table for easier access
pivot_df = df[df['model'] == model_type].pivot_table(
    index='modality', 
    columns='merged_project', 
    values=['mean_c_index', 'std_c_index'],
    aggfunc='first'
)

# Create the results dictionary
results = {}
for project in projects:
    results[project] = {}
    for modality in modalities:
        try:
            mean_val = pivot_df[('mean_c_index', project)][modality]
            std_val = pivot_df[('std_c_index', project)][modality]
            results[project][modality] = (mean_val, std_val)
        except:
            results[project][modality] = (np.nan, np.nan)

# Define better modality names
modality_names = {
    'clinical': 'Clinical',
    'pathology': 'Pathology',
    'radiology': 'Radiology',
    'molecular': 'Molecular',
    'wsi': 'WSI',
    'concat': 'Multimodal (Concat)',
    'mean_pool': 'Multimodal (Mean Pool)',
    'kronecker': 'Multimodal (Kronecker)'
}

# Generate Markdown table
print("## Survival Analysis C-Index Results (Cox Model)\n")
print("| Modality | " + " | ".join(projects) + " |")
print("|" + "-"*10 + "|" + "|".join(["-"*20 for _ in projects]) + "|")

for modality in modalities:
    row = f"| {modality_names[modality]:<20} |"
    for project in projects:
        mean_val, std_val = results[project][modality]
        if not np.isnan(mean_val):
            row += f" {mean_val:.3f} ± {std_val:.3f} |"
        else:
            row += " - |"
    print(row)

# Generate LaTeX table
print("\n\n## LaTeX Table\n")
print("```latex")
print("\\begin{table}[htbp]")
print("\\centering")
print("\\caption{Survival Analysis C-Index Results (Cox Model) for Different Modalities Across TCGA Projects}")
print("\\label{tab:survival_cindex}")
print("\\resizebox{\\textwidth}{!}{%")
print("\\begin{tabular}{l" + "c" * len(projects) + "}")
print("\\toprule")
print("Modality & " + " & ".join([p.replace("TCGA-", "") for p in projects]) + " \\\\")
print("\\midrule")

# Group modalities
print("\\multicolumn{" + str(len(projects) + 1) + "}{l}{\\textit{Unimodal}} \\\\")
for modality in ['clinical', 'pathology', 'radiology', 'molecular', 'wsi']:
    row = f"{modality_names[modality]}"
    for project in projects:
        mean_val, std_val = results[project][modality]
        if not np.isnan(mean_val):
            row += f" & ${mean_val:.3f} \\pm {std_val:.3f}$"
        else:
            row += " & -"
    row += " \\\\"
    print(row)

print("\\midrule")
print("\\multicolumn{" + str(len(projects) + 1) + "}{l}{\\textit{Multimodal}} \\\\")
for modality in ['concat', 'mean_pool', 'kronecker']:
    row = f"{modality_names[modality].replace('Multimodal ', '')}"
    for project in projects:
        mean_val, std_val = results[project][modality]
        if not np.isnan(mean_val):
            # Bold the best multimodal result for each project
            best_multi = max([results[project][m][0] for m in ['concat', 'mean_pool', 'kronecker'] 
                            if not np.isnan(results[project][m][0])])
            if mean_val == best_multi:
                row += f" & $\\mathbf{{{mean_val:.3f} \\pm {std_val:.3f}}}$"
            else:
                row += f" & ${mean_val:.3f} \\pm {std_val:.3f}$"
        else:
            row += " & -"
    row += " \\\\"
    print(row)

print("\\bottomrule")
print("\\end{tabular}%")
print("}")
print("\\end{table}")
print("```")

# Generate a summary statistics section
print("\n\n## Summary Statistics\n")

# Find best performing modality for each project
print("### Best Performing Modality by Project (Cox Model)\n")
for project in projects:
    best_modality = None
    best_score = -1
    for modality in modalities:
        mean_val, _ = results[project][modality]
        if not np.isnan(mean_val) and mean_val > best_score:
            best_score = mean_val
            best_modality = modality
    print(f"- **{project}**: {modality_names[best_modality]} (C-index: {best_score:.3f})")

# Calculate average improvement of multimodal over best unimodal
print("\n### Multimodal vs Best Unimodal Performance\n")
for project in projects:
    # Find best unimodal
    best_uni = max([results[project][m][0] for m in ['clinical', 'pathology', 'radiology', 'molecular', 'wsi'] 
                    if not np.isnan(results[project][m][0])])
    # Find best multimodal
    best_multi = max([results[project][m][0] for m in ['concat', 'mean_pool', 'kronecker'] 
                     if not np.isnan(results[project][m][0])])
    improvement = ((best_multi - best_uni) / best_uni) * 100
    print(f"- **{project}**: {improvement:+.1f}% ({'improvement' if improvement > 0 else 'degradation'})")

# Create a simplified table showing only the best model for each modality type
print("\n\n## Simplified Best Performance Table\n")
print("| Project | Best Unimodal | C-Index | Best Multimodal | C-Index |")
print("|---------|---------------|---------|-----------------|---------|")

for project in projects:
    # Find best unimodal
    best_uni_modality = None
    best_uni_score = -1
    for modality in ['clinical', 'pathology', 'radiology', 'molecular', 'wsi']:
        mean_val, std_val = results[project][modality]
        if not np.isnan(mean_val) and mean_val > best_uni_score:
            best_uni_score = mean_val
            best_uni_std = std_val
            best_uni_modality = modality
    
    # Find best multimodal
    best_multi_modality = None
    best_multi_score = -1
    for modality in ['concat', 'mean_pool', 'kronecker']:
        mean_val, std_val = results[project][modality]
        if not np.isnan(mean_val) and mean_val > best_multi_score:
            best_multi_score = mean_val
            best_multi_std = std_val
            best_multi_modality = modality
    
    print(f"| {project} | {modality_names[best_uni_modality]} | "
          f"{best_uni_score:.3f} ± {best_uni_std:.3f} | "
          f"{modality_names[best_multi_modality].replace('Multimodal ', '')} | "
          f"{best_multi_score:.3f} ± {best_multi_std:.3f} |")