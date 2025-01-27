import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ncaa_analysis
import local_analysis

ncaa_analysis_df = ncaa_analysis.create_ncaa_analysis_df()
local_analysis_df = local_analysis.create_local_analysis_df()

###---------------------------------------NCAA VS LOCAL CORRELATION DIFFERENCES START---------------------------------------###
def compare_correlation_matrices():
    """
    Compares correlation matrices between NCAA and local data to identify differences
    in relationships between metrics. Subtracts local from NCAA.
    
    Returns:
        pd.DataFrame: Matrix of correlation differences (NCAA - Local)
    """
    # Get correlation matrices from analysis files
    ncaa_corr = ncaa_analysis.plot_correlation_matrix(ncaa_analysis_df)
    local_corr = local_analysis.plot_correlation_matrix(local_analysis_df)
    
    # Calculate differences (NCAA - Local)
    diff_matrix = (ncaa_corr - local_corr).round(3)
    
    # Create heatmap of differences
    plt.figure(figsize=(12, 10))
    sns.heatmap(diff_matrix,
                annot=True,
                cmap='RdBu',
                center=0,
                fmt='.3f',
                square=True)
    
    plt.title('Correlation Differences (NCAA - Local)')
    plt.tight_layout()
    plt.savefig('ncaa_vs_local_plots/ncaa_vs_local_corr_differences.png')
    plt.close()
        
    return diff_matrix

#correlation_differences = compare_correlation_matrices()

def compare_group_correlation_matrices(group_type, group):
    """
    Compares correlation matrices between NCAA and local data for a specific group
    (stroke or coach) to identify differences in relationships between metrics.
    Subtracts local from NCAA.
    
    Args:
        group_type (str): Type of group to analyze ('Stroke' or 'Coach')
        group (str): Specific group name (e.g. 'Free', 'Fly', 'Herbie', etc.)
        
    Returns:
        pd.DataFrame: Matrix of correlation differences (NCAA - Local) for the group
    """
    # Filter local data for group
    local_group = local_analysis_df[local_analysis_df[group_type] == group]
    
    # Get correlation matrices
    ncaa_corr = ncaa_analysis.plot_correlation_matrix(ncaa_analysis_df)
    local_group_corr = local_analysis.plot_correlation_matrix(local_group)
    
    # Calculate differences (NCAA - Local group)
    diff_matrix = (ncaa_corr - local_group_corr).round(3)
    
    # Create heatmap of differences
    plt.figure(figsize=(12, 10))
    sns.heatmap(diff_matrix,
                annot=True, 
                cmap='RdBu',
                center=0,
                fmt='.3f',
                square=True)
    
    plt.title(f'Correlation Differences (NCAA - {group} {group_type})')
    plt.tight_layout()
    plt.savefig(f'ncaa_vs_local_plots/ncaa_vs_{group.lower()}_{group_type.lower()}_corr_differences.png')
    plt.close()
        
    return diff_matrix

# All groups called below
#group_vs_ncaa_correlation_differences = compare_group_correlation_matrices('Stroke', 'Free')

def analyze_all_groups():
    """
    Analyzes correlation differences for all stroke and coach groups by calling
    compare_group_correlation_matrices() for each group.
    """
    strokes = ['Free', 'Fly', 'Back', 'Breast']
    coaches = ['Herbie', 'Logan', 'Corey', 'Salo']

    for stroke in strokes:
        correlation_differences = compare_group_correlation_matrices('Stroke', stroke)
   
    for coach in coaches:
        correlation_differences = compare_group_correlation_matrices('Coach', coach)

#analyze_all_groups()
###---------------------------------------NCAA VS LOCAL CORRELATION DIFFERENCES END---------------------------------------###

###---------------------------------------NCAA VS LOCAL OBJECTIVE METRICS START---------------------------------------###

