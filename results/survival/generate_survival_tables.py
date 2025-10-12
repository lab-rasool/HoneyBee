#!/usr/bin/env python3
"""
Generate markdown and LaTeX tables for survival analysis C-index results.
Creates 7 separate transposed tables, with multimodal methods split for better fit.
"""

import pandas as pd
import numpy as np

# Load the survival analysis summary
df = pd.read_csv('survival_analysis_summary_v3.csv')

# Get unique projects and models
projects = sorted(df['merged_project'].unique())
models = ['cox', 'rsf', 'deepsurv']

# Define modality groups
modality_groups = {
    'Clinical Features': ['clinical'],
    'Pathology Report Features': ['pathology'],
    'Radiology Features': ['radiology'],
    'Molecular Features': ['molecular'],
    'WSI Features': ['wsi'],
    'Multimodal Features (Concatenation & Mean Pooling)': ['concat', 'mean_pool'],
    'Multimodal Features (Kronecker Product)': ['kronecker']
}

# Model display names
model_names = {
    'cox': 'Cox',
    'rsf': 'RSF',
    'deepsurv': 'DeepSurv'
}

# Multimodal method names
multimodal_names = {
    'concat': 'Concatenation',
    'mean_pool': 'Mean Pooling',
    'kronecker': 'Kronecker Product'
}

def format_value(mean_val, std_val):
    """Format C-index value with standard deviation."""
    if not np.isnan(mean_val):
        return f"{mean_val:.3f} ± {std_val:.3f}"
    else:
        return "-"

def generate_latex_table(feature_name, projects_subset, results_dict, table_num=None, is_multimodal=False, modalities=None):
    """Generate LaTeX table for a single feature category."""
    print(f"\n% ===== {feature_name.upper()} =====")
    
    # Filter out projects where all values are "-"
    filtered_projects = []
    for project in projects_subset:
        has_data = False
        if is_multimodal:
            # Check if any multimodal method has data
            for method in modalities:
                for model in models:
                    if results_dict.get(project, {}).get(method, {}).get(model, "-") != "-":
                        has_data = True
                        break
                if has_data:
                    break
        else:
            for model in models:
                if results_dict.get(project, {}).get(model, "-") != "-":
                    has_data = True
                    break
        if has_data:
            filtered_projects.append(project)
    
    # If no data for this feature, skip the table
    if not filtered_projects:
        print(f"% No data available for {feature_name}")
        return
    
    caption = f"Survival analysis results - {feature_name}. C-index values are reported as mean ± standard deviation across 5-fold cross-validation."
    
    label = f"survival_{feature_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('&', 'and')}"
    if table_num:
        label += f"_{table_num}"
    
    if is_multimodal:
        # Determine which methods to show
        methods_to_show = modalities
        
        if len(methods_to_show) == 2:  # Concat + Mean Pool
            print(f"\\begin{{table}}[htbp]")
            print(f"    \\centering")
            print(f"    \\caption{{{caption}}}")
            print(f"    \\label{{tab:{label}}}")
            print(f"    \\begin{{tabular}}{{@{{}}l|ccc|ccc@{{}}}}")
            print(f"        \\toprule")
            print(f"        & \\multicolumn{{3}}{{c|}}{{\\textbf{{Concatenation}}}} & \\multicolumn{{3}}{{c}}{{\\textbf{{Mean Pooling}}}} \\\\")
            print(f"        \\cmidrule(lr){{2-4}} \\cmidrule(lr){{5-7}}")
            print(f"        \\textbf{{Cancer}} & \\textbf{{Cox}} & \\textbf{{RSF}} & \\textbf{{DeepSurv}} & \\textbf{{Cox}} & \\textbf{{RSF}} & \\textbf{{DeepSurv}} \\\\")
            print(f"        \\midrule")
            
            for project in filtered_projects:
                project_short = project.replace('TCGA-', '')
                row = f"        {project_short:<4}"
                
                # Add values for each multimodal method
                for method_key in methods_to_show:
                    for model in models:
                        value = results_dict.get(project, {}).get(method_key, {}).get(model, "-")
                        row += f" & {value}"
                
                row += " \\\\"
                print(row)
            
            print(f"        \\bottomrule")
            print(f"    \\end{{tabular}}")
            print(f"\\end{{table}}")
        
        elif len(methods_to_show) == 1:  # Kronecker only
            print(f"\\begin{{table}}[htbp]")
            print(f"    \\centering")
            print(f"    \\caption{{{caption}}}")
            print(f"    \\label{{tab:{label}}}")
            print(f"    \\begin{{tabular}}{{@{{}}lccc@{{}}}}")
            print(f"        \\toprule")
            print(f"        \\textbf{{Cancer Type}} & \\textbf{{Cox}} & \\textbf{{RSF}} & \\textbf{{DeepSurv}} \\\\")
            print(f"        \\midrule")
            
            for project in filtered_projects:
                project_short = project.replace('TCGA-', '')
                row = f"        {project_short:<4}"
                
                method_key = methods_to_show[0]
                for model in models:
                    value = results_dict.get(project, {}).get(method_key, {}).get(model, "-")
                    row += f" & {value}"
                
                row += " \\\\"
                print(row)
            
            print(f"        \\bottomrule")
            print(f"    \\end{{tabular}}")
            print(f"\\end{{table}}")
    else:
        # Regular single-modality table
        print(f"\\begin{{table}}[htbp]")
        print(f"    \\centering")
        print(f"    \\caption{{{caption}}}")
        print(f"    \\label{{tab:{label}}}")
        print(f"    \\begin{{tabular}}{{@{{}}lccc@{{}}}}")
        print(f"        \\toprule")
        print(f"        \\textbf{{Cancer Type}} & \\textbf{{Cox}} & \\textbf{{RSF}} & \\textbf{{DeepSurv}} \\\\")
        print(f"        \\midrule")
        
        for project in filtered_projects:
            project_short = project.replace('TCGA-', '')
            row = f"        {project_short:<4}"
            
            for model in models:
                value = results_dict.get(project, {}).get(model, "-")
                row += f" & {value}"
            
            row += " \\\\"
            print(row)
        
        print(f"        \\bottomrule")
        print(f"    \\end{{tabular}}")
        print(f"\\end{{table}}")

def generate_markdown_table(feature_name, projects_subset, results_dict, is_multimodal=False, modalities=None):
    """Generate Markdown table for a single feature category."""
    if is_multimodal:
        print(f"\n### {feature_name}")
        
        # Filter projects with data
        filtered_projects = []
        for project in projects_subset:
            has_data = False
            for method in modalities:
                for model in models:
                    if results_dict.get(project, {}).get(method, {}).get(model, "-") != "-":
                        has_data = True
                        break
                if has_data:
                    break
            if has_data:
                filtered_projects.append(project)
        
        if not filtered_projects:
            print("*No data available for this feature*")
            return
        
        if len(modalities) == 2:  # Concat + Mean Pool
            # Create horizontal table with both methods
            print(f"\n| Cancer | Concatenation ||| Mean Pooling |||")
            print(f"|--------|---------------|---------------|---------------|---------------|---------------|---------------|")
            print(f"|        | Cox | RSF | DeepSurv | Cox | RSF | DeepSurv |")
            print(f"|--------|-----|-----|----------|-----|-----|----------|")
            
            for project in filtered_projects:
                project_short = project.replace('TCGA-', '')
                row = f"| {project_short:<6} |"
                
                # Add values for each multimodal method
                for method_key in modalities:
                    for model in models:
                        value = results_dict.get(project, {}).get(method_key, {}).get(model, "-")
                        # Shorten values for markdown table
                        if value != "-" and " ± " in value:
                            parts = value.split(" ± ")
                            value = f"{parts[0]}±{parts[1]}"
                        row += f" {value:<13} |"
                
                print(row)
        
        elif len(modalities) == 1:  # Kronecker only
            # Regular table format for single method
            print(f"\n| Cancer Type | Cox | RSF | DeepSurv |")
            print(f"|-------------|-----|-----|----------|")
            
            method_key = modalities[0]
            for project in filtered_projects:
                project_short = project.replace('TCGA-', '')
                row = f"| {project_short:<11} |"
                
                for model in models:
                    value = results_dict.get(project, {}).get(method_key, {}).get(model, "-")
                    row += f" {value:<15} |"
                
                print(row)
    else:
        # Regular single-modality table
        # Filter out projects where all values are "-"
        filtered_projects = []
        for project in projects_subset:
            has_data = False
            for model in models:
                if results_dict.get(project, {}).get(model, "-") != "-":
                    has_data = True
                    break
            if has_data:
                filtered_projects.append(project)
        
        # If no data for this feature, skip the table
        if not filtered_projects:
            print(f"\n### {feature_name}")
            print("*No data available for this feature*")
            return
        
        print(f"\n### {feature_name}")
        print(f"| Cancer Type | Cox | RSF | DeepSurv |")
        print(f"|-------------|-----|-----|----------|")
        
        for project in filtered_projects:
            project_short = project.replace('TCGA-', '')
            row = f"| {project_short:<11} |"
            
            for model in models:
                value = results_dict.get(project, {}).get(model, "-")
                row += f" {value:<15} |"
            
            print(row)

# Process each modality group
print("# Survival Analysis Results - Transposed Tables\n")
print("## Markdown Format\n")

for idx, (feature_name, modalities) in enumerate(modality_groups.items(), 1):
    # Get results for this feature category
    results_dict = {}
    
    is_multimodal = ('Multimodal' in feature_name)
    
    if is_multimodal:
        # Handle multimodal case - multiple modalities
        for project in projects:
            results_dict[project] = {}
            
            for modality in modalities:
                results_dict[project][modality] = {}
                
                for model in models:
                    # Filter data
                    mask = (df['merged_project'] == project) & \
                           (df['modality'] == modality) & \
                           (df['model'] == model)
                    
                    subset = df[mask]
                    
                    if not subset.empty:
                        mean_val = subset['mean_c_index'].iloc[0]
                        std_val = subset['std_c_index'].iloc[0]
                        results_dict[project][modality][model] = format_value(mean_val, std_val)
                    else:
                        results_dict[project][modality][model] = "-"
    else:
        # Handle single modality case
        for project in projects:
            results_dict[project] = {}
            
            for model in models:
                # For this feature category, we'll use the first (and only) modality in the list
                modality = modalities[0]
                
                # Filter data
                mask = (df['merged_project'] == project) & \
                       (df['modality'] == modality) & \
                       (df['model'] == model)
                
                subset = df[mask]
                
                if not subset.empty:
                    mean_val = subset['mean_c_index'].iloc[0]
                    std_val = subset['std_c_index'].iloc[0]
                    results_dict[project][model] = format_value(mean_val, std_val)
                else:
                    results_dict[project][model] = "-"
    
    # Generate markdown table
    generate_markdown_table(feature_name, projects, results_dict, is_multimodal, modalities)

# Generate LaTeX tables
print("\n\n## LaTeX Format\n")
print("```latex")

for idx, (feature_name, modalities) in enumerate(modality_groups.items(), 1):
    # Get results for this feature category
    results_dict = {}
    
    is_multimodal = ('Multimodal' in feature_name)
    
    if is_multimodal:
        # Handle multimodal case - multiple modalities
        for project in projects:
            results_dict[project] = {}
            
            for modality in modalities:
                results_dict[project][modality] = {}
                
                for model in models:
                    # Filter data
                    mask = (df['merged_project'] == project) & \
                           (df['modality'] == modality) & \
                           (df['model'] == model)
                    
                    subset = df[mask]
                    
                    if not subset.empty:
                        mean_val = subset['mean_c_index'].iloc[0]
                        std_val = subset['std_c_index'].iloc[0]
                        results_dict[project][modality][model] = format_value(mean_val, std_val)
                    else:
                        results_dict[project][modality][model] = "-"
    else:
        # Handle single modality case
        for project in projects:
            results_dict[project] = {}
            
            for model in models:
                # For this feature category, we'll use the first (and only) modality in the list
                modality = modalities[0]
                
                # Filter data
                mask = (df['merged_project'] == project) & \
                       (df['modality'] == modality) & \
                       (df['model'] == model)
                
                subset = df[mask]
                
                if not subset.empty:
                    mean_val = subset['mean_c_index'].iloc[0]
                    std_val = subset['std_c_index'].iloc[0]
                    results_dict[project][model] = format_value(mean_val, std_val)
                else:
                    results_dict[project][model] = "-"
    
    # Generate LaTeX table
    generate_latex_table(feature_name, projects, results_dict, idx, is_multimodal, modalities)

print("```")

# Generate summary statistics
print("\n\n## Summary Statistics\n")

# Find best performing method for each cancer type and feature
print("### Best Performing Method by Cancer Type and Feature\n")

for feature_name, modalities in modality_groups.items():
    print(f"\n**{feature_name}:**")
    
    if 'Multimodal' in feature_name:
        # For multimodal, show best across all methods
        for project in projects[:5]:  # Show first 5 as examples
            project_short = project.replace('TCGA-', '')
            best_model = None
            best_method = None
            best_score = -1
            
            for modality in modalities:
                for model in models:
                    mask = (df['merged_project'] == project) & \
                           (df['modality'] == modality) & \
                           (df['model'] == model)
                    subset = df[mask]
                    
                    if not subset.empty:
                        mean_val = subset['mean_c_index'].iloc[0]
                        if not np.isnan(mean_val) and mean_val > best_score:
                            best_score = mean_val
                            best_model = model
                            best_method = modality
            
            if best_model:
                print(f"  - {project_short}: {multimodal_names[best_method]} + {model_names[best_model]} (C-index: {best_score:.3f})")
    else:
        modality = modalities[0]
        
        for project in projects[:5]:  # Show first 5 as examples
            project_short = project.replace('TCGA-', '')
            best_model = None
            best_score = -1
            
            for model in models:
                mask = (df['merged_project'] == project) & \
                       (df['modality'] == modality) & \
                       (df['model'] == model)
                subset = df[mask]
                
                if not subset.empty:
                    mean_val = subset['mean_c_index'].iloc[0]
                    if not np.isnan(mean_val) and mean_val > best_score:
                        best_score = mean_val
                        best_model = model
            
            if best_model:
                print(f"  - {project_short}: {model_names[best_model]} (C-index: {best_score:.3f})")