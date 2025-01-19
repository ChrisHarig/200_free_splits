import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os

# Read the CSV
splits_df = pd.read_csv('data/swim_splits.csv')
groups_df = pd.read_csv('data/group_stroke.csv')

# Get only numeric columns (excluding name column)
numeric_cols = splits_df.select_dtypes(include=['float64', 'int64']).columns

# Create analysis df with all original columns
analysis_df = splits_df.copy()

# Convert Final times from MM:SS.XX format to seconds
analysis_df['Final'] = analysis_df['Final'].apply(lambda x: float(x.split(':')[0])*60 + float(x.split(':')[1]))

# Calculate mean and std dev for each swimmer
split_cols = ['Split1', 'Split2', 'Split3', 'Split4']
analysis_df['mean'] = analysis_df[split_cols].mean(axis=1)
analysis_df['std_dev'] = analysis_df[split_cols].std(axis=1)

# Add all possible split differences
analysis_df['split4_3'] = analysis_df['Split4'] - analysis_df['Split3']  
analysis_df['split4_2'] = analysis_df['Split4'] - analysis_df['Split2']
analysis_df['split4_1'] = analysis_df['Split4'] - analysis_df['Split1']  
analysis_df['split3_2'] = analysis_df['Split3'] - analysis_df['Split2']
analysis_df['split3_1'] = analysis_df['Split3'] - analysis_df['Split1']
analysis_df['split2_1'] = analysis_df['Split2'] - analysis_df['Split1']  

# Calculate first 100 vs second 100 difference
analysis_df['first_100'] = analysis_df[['Split1', 'Split2']].sum(axis=1)
analysis_df['second_100'] = analysis_df[['Split3', 'Split4']].sum(axis=1)
analysis_df['hundred_diff'] = analysis_df['second_100'] - analysis_df['first_100']

# Add rankings based on time (faster times = better rank)
analysis_df['time_rank'] = analysis_df['Final'].rank()

# Add rankings based on consistency (lower std dev = better rank)
analysis_df['var_rank'] = analysis_df['std_dev'].rank()

# Add the group and stroke columns to the analysis_df
analysis_df = analysis_df.merge(groups_df[['Name', 'Group', 'Stroke']], on='Name', how='left')

def plot_consistency_vs_performance(analysis_df, show_names=False):
    """
    Creates a scatter plot comparing swimmer consistency (std dev) vs performance (final time).
    --->With the provided data, we find there is no relationship between consistency and performance.<---
    
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
        show_names (bool): Whether to show swimmer names as labels
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(analysis_df['std_dev'], analysis_df['Final'], alpha=0.6)
    plt.xlabel('Standard Deviation (Consistency)')
    plt.ylabel('Final Time')
    plt.title('Swimming Performance vs Consistency')
    
    if show_names:
        for idx, row in analysis_df.iterrows():
            plt.annotate(row['Name'], (row['std_dev'], row['Final']))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    correlation = analysis_df['std_dev'].corr(analysis_df['Final'])
    print(f"Correlation between consistency and final time: {correlation:.3f}")
    
    plt.savefig('team_plots/consistency_vs_performance.png')
    plt.close()

#plot_consistency_vs_performance(analysis_df)

def calculate_rank_correlations(analysis_df):
    """
    Calculates rank correlations between Consistency, Drop-off, and 100 Split Diff vs Performance across the whole team.
    We use rank to normalize the data as to avoid problems with absolute differences in splits weighting the correlation.
    --->With the provided data, we find there is almost no correlation (-.08 to .000)<---

    Args:
        analysis_df (pd.DataFrame): DataFrame containing swimming analysis data
        
    Returns:
        pd.DataFrame: DataFrame containing rank correlations between metrics
    """
    # Create a copy of the dataframe for rankings
    rank_df = analysis_df.copy()
    
    # Add rankings for key metrics
    rank_df['dropoff_rank'] = rank_df['split4_1'].rank()  # Lower drop-off = better rank
    rank_df['hundred_diff_rank'] = rank_df['hundred_diff'].rank()  # Lower difference = better rank

    # Calculate rank correlations
    rank_corr = pd.DataFrame({
        'metric': ['Consistency vs Performance', 'Drop-off vs Performance', '100 Split Diff vs Performance'],
        'correlation': [
            rank_df['var_rank'].corr(rank_df['time_rank'], method='spearman'),
            rank_df['dropoff_rank'].corr(rank_df['time_rank'], method='spearman'),
            rank_df['hundred_diff_rank'].corr(rank_df['time_rank'], method='spearman')
        ]
    })

    print("\nRank Correlations:")
    print(rank_corr)
    
    return rank_corr

#calculate_rank_correlations(analysis_df)

def plot_correlation_matrix(analysis_df):
    """
    Creates a correlation matrix heatmap for all split-related metrics.
    --->- No one split difference is correlated with final time
        - Both the fade from the initial pace to the third 50 (split3_1) 
        and the fade from the initial pace to the fourth 50 (split4_1) are 
        correlated with std_dev, but the third 50 accounts for more variance .94 > .90
        - split2_1 and split3_1 are highly correlated (.99), split3_1 and split3_2 are also highly correlated (.98), indicating 
        that one dropoff may account for the other, or there is a kind of physiological rule that forces this pattern.
        - IN PROGRESS<---
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
    """
    # Select relevant columns for correlation
    cols_to_correlate = [
        'Final', 'std_dev',
        'split4_3', 'split4_2', 'split4_1',
        'split3_2', 'split3_1', 'split2_1',
        'hundred_diff',
        'first_100', 'second_100'
    ]
    
    # Calculate correlation matrix
    corr_matrix = analysis_df[cols_to_correlate].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, 
                annot=True,          
                cmap='RdBu',         
                center=0,           
                fmt='.2f',          
                square=True)         
    
    plt.title('Correlation Matrix Team Wide')
    plt.tight_layout()
    plt.savefig('team_plots/correlation_matrix.png')
    plt.close()
    
    return corr_matrix

corr_matrix = plot_correlation_matrix(analysis_df)

def plot_group_correlation_matrix(analysis_df, group_name, include=True):
    """
    Creates a correlation matrix heatmap for split-related metrics, either including or excluding a specific group.
    Automatically detects if group_name exists in Stroke or Coach columns, and plots the data for that group if 
    include is True, or plots the data for all other groups if include is False.
    --->Results listed in next method<----
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
        group_name (str): Name of group to analyze (e.g. 'fly', 'free', 'Logan')
        include (bool): If True, only include specified group. If False, exclude specified group.
    """
    # Check if group exists in Stroke or Coach columns
    if group_name in analysis_df['Stroke'].unique():
        group_type = 'Stroke'
    elif group_name in analysis_df['Group'].unique():
        group_type = 'Group'
    else:
        raise ValueError(f"'{group_name}' not found in either Stroke or Coach columns")
        
    # Filter dataframe based on include/exclude flag
    if include:
        group_df = analysis_df[analysis_df[group_type] == group_name]
        suffix = f"{group_type.lower()}_{group_name}"
    else:
        group_df = analysis_df[analysis_df[group_type] != group_name]
        suffix = f"no_{group_type.lower()}_{group_name}"
    
    # Select relevant columns for correlation
    cols_to_correlate = [
        'Final', 'std_dev',
        'split4_3', 'split4_2', 'split4_1',
        'split3_2', 'split3_1', 'split2_1',
        'hundred_diff',
        'first_100', 'second_100'
    ]
    
    # Calculate correlation matrix
    group_corr = group_df[cols_to_correlate].corr()
    
    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(group_corr, 
                annot=True,          
                cmap='RdBu',         
                center=0,           
                fmt='.2f',          
                square=True)         
    
    plt.title(f'Correlation Matrix - {group_type}: {group_name}')
    plt.tight_layout()
    plt.savefig(f'team_plots/correlation_matrix_{suffix}.png')
    plt.close()

    return group_corr

# Skip to plot_all_group_correlations to plot all possible groups and strokes
#corr_matrix = plot_group_correlation_matrix(analysis_df, 'Logan', True)

def plot_all_group_correlations(analysis_df):
    """
    Creates correlation matrices for each unique group/stroke, both including and excluding each group.
    --->IN PROGRESS<----
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
    """
    # Get unique groups from both Stroke and Group columns
    strokes = analysis_df['Stroke'].unique()
    groups = analysis_df['Group'].unique()
    
    # Create correlation matrices for each group
    for group in strokes:
        # Include 
        plot_group_correlation_matrix(analysis_df, group, include=True)
        # Exclude 
        plot_group_correlation_matrix(analysis_df, group, include=False)
        
    for group in groups:
        # Include 
        plot_group_correlation_matrix(analysis_df, group, include=True)
        # Exclude 
        plot_group_correlation_matrix(analysis_df, group, include=False)

plot_all_group_correlations(analysis_df)

def analyze_group_differences(analysis_df, diff_threshold=0.15):
    """
    Analyzes how each group's correlations differ from the team average. Compares included groups vs team average.
    If a caertain groups corellation between two metrics is significantyl lower or higher that the team average,
    it could indicate the groups swimmers have a physiolgical weak spot or strength, or a race strategy weak spot or strength,
    that the team does not.
    --->IN PROGRESS<----
    
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
        diff_threshold (float, optional): Threshold for considering correlation differences meaningful. Defaults to 0.15.
    """
    # Get team-wide correlation matrix as baseline
    team_corr = plot_correlation_matrix(analysis_df)
    
    # Get unique groups from both Stroke and Group columns
    strokes = analysis_df['Stroke'].unique()
    groups = analysis_df['Group'].unique()
    
    # Set threshold 
    diff_threshold = diff_threshold
    
    # Store notable differences
    notable_diffs = []
    
    # Analyze each group type
    for group_type, group_list in [('Stroke', strokes), ('Group', groups)]:
        for group in group_list:
            # Get group correlation matrix
            group_df = analysis_df[analysis_df[group_type] == group]
            group_corr = plot_correlation_matrix(group_df)  
            
            # Calculate differences from team average
            diff_matrix = group_corr - team_corr
            
            # Find meaningful differences
            for i in range(len(diff_matrix.index)):
                for j in range(i+1, len(diff_matrix.columns)):
                    diff = diff_matrix.iloc[i,j]
                    if abs(diff) >= diff_threshold:
                        notable_diffs.append({
                            'group_type': group_type,
                            'group': group,
                            'metric1': diff_matrix.index[i],
                            'metric2': diff_matrix.columns[j],
                            'team_corr': team_corr.iloc[i,j],
                            'group_corr': group_corr.iloc[i,j],
                            'difference': diff
                        })
    
    # Convert to DataFrame and sort by absolute difference
    results_df = pd.DataFrame(notable_diffs)
    results_df['abs_diff'] = results_df['difference'].abs()
    results_df = results_df.sort_values('abs_diff', ascending=False)
    
    print("\nNotable Correlation Differences from Team Average:")
    print(results_df.to_string(index=False))
    
    return results_df

group_differences = analyze_group_differences(analysis_df)
print(group_differences)

