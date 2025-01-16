import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

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
    Calculates rank correlations between Consistency, Drop-off, and 100 Split Diff vs Performance.
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
    --->In Progress<---
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
                annot=True,          # Show correlation values
                cmap='RdBu',         # Red-Blue diverging colormap
                center=0,            # Center the colormap at 0
                fmt='.2f',           # Round to 2 decimal places
                square=True)         # Make cells square
    
    plt.title('Correlation Matrix of Swimming Metrics')
    plt.tight_layout()
    plt.show()
    
    return corr_matrix

#corr_matrix = plot_correlation_matrix(analysis_df)



