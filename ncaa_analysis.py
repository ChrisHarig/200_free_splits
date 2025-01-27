import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###---------------------------------------DATA PREP START---------------------------------------###

def create_ncaa_analysis_df():
    """
    Creates analysis DataFrame from NCAA data files.
    Combines multiple years of data and adds year to swimmer names to differentiate repeat swimmers.
    
    Returns:
        pd.DataFrame: Combined NCAA data with years appended to names
    """
    # Initialize empty list to store DataFrames
    dfs = []
    
    # Loop through files in data/ncaa directory
    ncaa_dir = 'data/ncaa_results'
    for filename in os.listdir(ncaa_dir):
        # Read the CSV
        file_path = os.path.join(ncaa_dir, filename)
        df = pd.read_csv(file_path)
        
        # Extract year from filename (assuming format'2022.csv')
        year = filename.split('.')[0]
        
        # Append year to names
        df['Name'] = df['Name'] + " '" + year[-2:]
        
        dfs.append(df)
    
    # Combine all DataFrames
    ncaa_analysis_df = pd.concat(dfs, ignore_index=True)
    
    # Convert Final times from MM:SS.XX format to seconds
    ncaa_analysis_df['Final'] = ncaa_analysis_df['Final'].apply(lambda x: float(x.split(':')[0])*60 + float(x.split(':')[1])).round(2)
    
    # Calculate mean and std dev for each swimmer
    split_cols = ['Split1', 'Split2', 'Split3', 'Split4']
    ncaa_analysis_df['mean'] = ncaa_analysis_df[split_cols].mean(axis=1).round(2)
    ncaa_analysis_df['std_dev'] = ncaa_analysis_df[split_cols].std(axis=1)
    
    # Add all possible split differences
    ncaa_analysis_df['split4_3'] = (ncaa_analysis_df['Split4'] - ncaa_analysis_df['Split3']).round(2)
    ncaa_analysis_df['split4_2'] = (ncaa_analysis_df['Split4'] - ncaa_analysis_df['Split2']).round(2)
    ncaa_analysis_df['split4_1'] = (ncaa_analysis_df['Split4'] - ncaa_analysis_df['Split1']).round(2)
    ncaa_analysis_df['split3_2'] = (ncaa_analysis_df['Split3'] - ncaa_analysis_df['Split2']).round(2)
    ncaa_analysis_df['split3_1'] = (ncaa_analysis_df['Split3'] - ncaa_analysis_df['Split1']).round(2)
    ncaa_analysis_df['split2_1'] = (ncaa_analysis_df['Split2'] - ncaa_analysis_df['Split1']).round(2)
    
    # Calculate first 100 vs second 100 difference
    ncaa_analysis_df['first_100'] = ncaa_analysis_df[['Split1', 'Split2']].sum(axis=1).round(2)
    ncaa_analysis_df['second_100'] = ncaa_analysis_df[['Split3', 'Split4']].sum(axis=1).round(2)
    ncaa_analysis_df['hundred_diff'] = (ncaa_analysis_df['second_100'] - ncaa_analysis_df['first_100']).round(2)
    
    # Add rankings based on time (faster times = better rank)
    ncaa_analysis_df['time_rank'] = ncaa_analysis_df['Final'].rank()
    
    # Add rankings based on consistency (lower std dev = better rank)
    ncaa_analysis_df['var_rank'] = ncaa_analysis_df['std_dev'].rank()
    
    return ncaa_analysis_df

ncaa_analysis_df = create_ncaa_analysis_df()

###---------------------------------------DATA PREP END---------------------------------------###

###---------------------------------------VARIANCE ANALYSIS START---------------------------------------###

def plot_consistency_vs_performance(analysis_df, show_names=False):
    """
    Creates a scatter plot comparing swimmer consistency (std dev) vs performance (final time).
    --->No recognizable relationship between consistency and performance found<---

    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
        show_names (bool): Whether to show swimmer names as labels
    """
    plt.figure(figsize=(10,6))
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
    
    plt.show()
    plt.savefig('ncaa_plots/consistency_vs_performance.png')
    plt.close()

#plot_consistency_vs_performance(ncaa_analysis_df, show_names=False)

def calculate_rank_correlations(analysis_df):
    """
    Calculates rank correlations between Consistency, Drop-off, and 100 Split Diff vs Performance across NCAA data.
    We use rank to normalize the data as to avoid problems with absolute differences in splits weighting the correlation.
    --->With the provided data, we find there is almost no correlation (-.05 to .00)<---

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

#calculate_rank_correlations(ncaa_analysis_df)

def plot_correlation_matrix(analysis_df):
    """
    Creates a correlation matrix heatmap for all split-related metrics.
    --->Similar to the patterns found in local_analysis.py. Comparisons found in local_vs_ncaa.py<---
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
    plt.savefig('ncaa_plots/ncaa_correlation_matrix.png')
    plt.close()
    
    return corr_matrix

#plot_correlation_matrix(ncaa_analysis_df)

###---------------------------------------VARIANCE ANALYSIS END---------------------------------------###

###---------------------------------------SPLIT DROP OFFS START---------------------------------------###
def analyze_split_dropoffs(analysis_df):
    """
    Calculates summary statistics (mean, max, min) for all split dropoff metrics.
    
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data with split metrics
    
    Returns:
        pd.DataFrame: Summary statistics for each split dropoff metric
    """
    # Define split dropoff columns
    split_cols = ['split4_3', 'split4_2', 'split4_1', 'split3_2', 'split3_1', 'split2_1']
    
    # Calculate statistics
    stats = {
        'Mean': analysis_df[split_cols].mean(),
        'Max': analysis_df[split_cols].max(),
        'Min': analysis_df[split_cols].min()
    }
    
    # Convert to DataFrame for nice formatting
    stats_df = pd.DataFrame(stats).round(3)

    # Print full DataFrame
    print("\nSplit Dropoff Statistics:")
    print(stats_df.to_string())
    
    return stats_df

split_stats = analyze_split_dropoffs(ncaa_analysis_df)

###---------------------------------------SPLIT DROP OFFS END---------------------------------------###

