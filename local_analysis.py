import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import os

###---------------------------------------DATA PREP START---------------------------------------###

def create_local_analysis_df():
    """
    Creates analysis DataFrame with split times, calculated metrics, and group data.
    Processes raw data into format needed for analysis functions.
    
    Returns:
        pd.DataFrame: Analysis-ready DataFrame with calculated metrics
    """
    # Read the CSV
    splits_df = pd.read_csv('data/swim_splits.csv')
    groups_df = pd.read_csv('data/group_stroke.csv')

    # Get only numeric columns (excluding name column)
    numeric_cols = splits_df.select_dtypes(include=['float64', 'int64']).columns

    # Create analysis df with all original columns
    local_analysis_df = splits_df.copy()

    # Convert Final times from MM:SS.XX format to seconds
    local_analysis_df['Final'] = local_analysis_df['Final'].apply(lambda x: float(x.split(':')[0])*60 + float(x.split(':')[1]))

    # Calculate mean and std dev for each swimmer
    split_cols = ['Split1', 'Split2', 'Split3', 'Split4']
    local_analysis_df['mean'] = local_analysis_df[split_cols].mean(axis=1).round(2)
    local_analysis_df['std_dev'] = local_analysis_df[split_cols].std(axis=1)

    # Add all possible split differences
    local_analysis_df['split4_3'] = local_analysis_df['Split4'] - local_analysis_df['Split3'].round(2)
    local_analysis_df['split4_2'] = local_analysis_df['Split4'] - local_analysis_df['Split2'].round(2)
    local_analysis_df['split4_1'] = local_analysis_df['Split4'] - local_analysis_df['Split1'].round(2)  
    local_analysis_df['split3_2'] = local_analysis_df['Split3'] - local_analysis_df['Split2'].round(2)
    local_analysis_df['split3_1'] = local_analysis_df['Split3'] - local_analysis_df['Split1'].round(2)
    local_analysis_df['split2_1'] = local_analysis_df['Split2'] - local_analysis_df['Split1'].round(2)

    # Calculate first 100 vs second 100 difference
    local_analysis_df['first_100'] = local_analysis_df[['Split1', 'Split2']].sum(axis=1)
    local_analysis_df['second_100'] = local_analysis_df[['Split3', 'Split4']].sum(axis=1).round(2)
    local_analysis_df['hundred_diff'] = (local_analysis_df['second_100'] - local_analysis_df['first_100']).round(2)

    # Add rankings based on time (faster times = better rank)
    local_analysis_df['time_rank'] = local_analysis_df['Final'].rank()

    # Add rankings based on consistency (lower std dev = better rank)
    local_analysis_df['var_rank'] = local_analysis_df['std_dev'].rank()

    # Add the group and stroke columns to the analysis_df
    local_analysis_df = local_analysis_df.merge(groups_df[['Name', 'Coach', 'Stroke']], on='Name', how='left')
    
    return local_analysis_df

local_analysis_df = create_local_analysis_df()

###---------------------------------------DATA PREP END---------------------------------------###

###---------------------------------------TEAM WIDE ANALYSIS START---------------------------------------###

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
    
    plt.show()
    plt.savefig('team_plots/consistency_vs_performance.png')
    plt.close()

#plot_consistency_vs_performance(local_analysis_df)

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

#calculate_rank_correlations(local_analysis_df)

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
    plt.show()

    plt.savefig('team_plots/correlation_matrix.png')
    plt.close()
    
    return corr_matrix

#corr_matrix = plot_correlation_matrix(local_analysis_df)

###---------------------------------------TEAM WIDE ANALYSIS END---------------------------------------###

###---------------------------------------GROUPED ANALYSIS START---------------------------------------###

def analyze_group_value_differences(analysis_df):
    """
    Analyzes how each group's averages differ from the team averages for key metrics.
    
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
        
    Returns:
        pd.DataFrame: DataFrame containing differences between group averages and team averages
    """
    # Metrics to analyze
    metrics = [
        'Final', 'std_dev', 
        'split4_3', 'split4_2', 'split4_1',
        'split3_2', 'split3_1', 'split2_1',
        'hundred_diff',
        'first_100', 'second_100'
    ]
    
    # Calculate team averages
    team_means = analysis_df[metrics].mean()
    
    # Initialize list to store results
    results = []
    
    # Check if both Stroke and Coach groupings exist
    group_columns = []
    if 'Stroke' in analysis_df.columns:
        group_columns.append('Stroke')
    if 'Coach' in analysis_df.columns:
        group_columns.append('Coach')
        
    # Calculate differences for each group
    for group_col in group_columns:
        for group in analysis_df[group_col].unique():
            group_data = analysis_df[analysis_df[group_col] == group]
            group_means = group_data[metrics].mean()
            
            # Calculate absolute and percentage differences
            abs_diff = group_means - team_means
            pct_diff = (group_means - team_means) / team_means * 100
            
            # Create row with group info and differences
            row = {
                'Group_Type': group_col,
                'Group': group
            }
            
            # Add absolute and percentage differences for each metric
            for metric in metrics:
                row[f'{metric}_abs'] = abs_diff[metric]
                row[f'{metric}_pct'] = pct_diff[metric]
            
            results.append(row)
    
    # Create DataFrame from results and set column order
    diff_df = pd.DataFrame(results)
    
    # Reorder columns to put group info first, then pairs of abs/pct differences
    cols = ['Group_Type', 'Group']
    for metric in metrics:
        cols.extend([f'{metric}_abs', f'{metric}_pct'])
    
    diff_df = diff_df[cols]
    
    # Save to CSV 
    diff_df.to_csv('team_plots/group_metric_differences.csv', index=False)

    # Print full DataFrame
    print("\nGroup Metric Differences:")
    print(diff_df.to_string(index=False))
    
    return diff_df

#group_value_differences = analyze_group_value_differences(local_analysis_df)

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
    elif group_name in analysis_df['Coach'].unique():
        group_type = 'Coach'
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
#corr_matrix = plot_group_correlation_matrix(local_analysis_df, 'Logan', True)

def plot_all_group_correlations(analysis_df):
    """
    Creates correlation matrices for each unique group/stroke, both including and excluding each group.
    --->IN PROGRESS<----
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
    """
    # Get unique groups from both Stroke and Coach columns
    strokes = analysis_df['Stroke'].unique()
    coaches = analysis_df['Coach'].unique()
    
    # Create correlation matrices for each groupCreate another method that does the same thing as analyze_group_differences but for differences in the values, not correlations. So for each group, if there is a large difference in their average split4_1 or std_deviation and so on, record that
    for group in strokes:
        # Include 
        plot_group_correlation_matrix(analysis_df, group, include=True)
        # Exclude 
        plot_group_correlation_matrix(analysis_df, group, include=False)
        
    for coach in coaches:
        # Include 
        plot_group_correlation_matrix(analysis_df, coach, include=True)
        # Exclude 
        plot_group_correlation_matrix(analysis_df, coach, include=False)

#plot_all_group_correlations(local_analysis_df)

def analyze_group_correlation_differences(analysis_df, diff_threshold=0.15):
    """
    Analyzes how each group's correlations differ from the team average. Compares included groups vs team average.
    If a certain groups corellation between two metrics is significantly lower or higher that the team average,
    it indicates the relationship between two metrics is stronger or weaker for the selected subset of swimmers.
    --->IN PROGRESS<----
    
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data
        diff_threshold (float, optional): Threshold for considering correlation differences meaningful. Defaults to 0.15.
    """
    # Get team-wide correlation matrix as baseline
    team_corr = plot_correlation_matrix(analysis_df)
    
    # Get unique groups from both Stroke and Coach columns
    strokes = analysis_df['Stroke'].unique()
    coaches = analysis_df['Coach'].unique()
    
    # Set threshold 
    diff_threshold = diff_threshold
    
    # Store notable differences
    notable_diffs = []
    
    # Analyze each group type
    for group_type, group_list in [('Stroke', strokes), ('Coach', coaches)]:
        for group in group_list:
            # Get group correlation matrix
            group_df = analysis_df[analysis_df[group_type] == group]
            group_corr = plot_correlation_matrix(group_df)  
            
            # Calculate differences from team average
            diff_matrix = group_corr - team_corr
            
            # Find and record meaningful differences
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
    
    # Save results to CSV
    results_df.to_csv('team_plots/notable_group_correlation_differences.csv', index=False)
    
    return results_df

#group_differences = analyze_group_correlation_differences(local_analysis_df, .30)
#print(group_differences.to_string(index=False))

###---------------------------------------GROUPED ANALYSIS END---------------------------------------###

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

split_stats = analyze_split_dropoffs(local_analysis_df)

def analyze_group_split_dropoffs(analysis_df, group_type):
    """
    Calculates summary statistics (mean, max, min) for all split dropoff metrics by group.
    
    Args:
        analysis_df (pd.DataFrame): DataFrame containing the analysis data with split metrics
        group_type (str): Column name to group by (e.g. 'Stroke' or 'Coach')
    
    Returns:
        dict: Dictionary mapping each group to its summary statistics DataFrame
    """
    # Define split dropoff columns
    split_cols = ['split4_3', 'split4_2', 'split4_1', 'split3_2', 'split3_1', 'split2_1']
    
    # Get unique groups
    groups = analysis_df[group_type].unique()
    
    # Store results for each group
    group_stats = {}
    
    for group in groups:
        # Filter data for this group
        group_df = analysis_df[analysis_df[group_type] == group]
        
        # Calculate statistics for this group
        stats = {
            'Mean': group_df[split_cols].mean(),
            'Max': group_df[split_cols].max(),
            'Min': group_df[split_cols].min()
        }
        
        # Convert to DataFrame and round
        stats_df = pd.DataFrame(stats).round(3)
        
        # Store in results dictionary
        group_stats[group] = stats_df
        
        # Print results for this group
        print(f"\nSplit Dropoff Statistics for {group} {group_type}:")
        print(stats_df.to_string())

        # Save to CSV
        filename = f'local_analysis_stats/{group_type.lower()}_{group.lower()}_split_stats.csv'
        stats_df.to_csv(filename)
    
    return group_stats

# Method below calculates for each group
#stroke_stats = analyze_group_split_dropoffs(local_analysis_df, 'Stroke')
#coach_stats = analyze_group_split_dropoffs(local_analysis_df, 'Coach')

def analyze_all_group_split_dropoffs():
    """
    Analyzes split dropoff statistics for all stroke and coach groups by calling
    analyze_group_split_dropoffs() for each group type.
    """
    group_types = ['Stroke', 'Coach']
    
    for group_type in group_types:
        print(f"\n{'-'*20} {group_type} Analysis {'-'*20}")
        group_stats = analyze_group_split_dropoffs(local_analysis_df, group_type)

#analyze_all_group_split_dropoffs()
