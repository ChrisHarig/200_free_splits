import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV
splits_df = pd.read_csv('data/swim_splits.csv')
groups_df = pd.read_csv('data/group_stroke.csv')

# Get only numeric columns (excluding name column)
numeric_cols = splits_df.select_dtypes(include=['float64', 'int64']).columns

# Create analysis df with all original columns
analysis_df = splits_df.copy()

# Convert Final times from MM:SS.XX format to seconds
analysis_df['Final'] = analysis_df['Final'].apply(lambda x: float(x.split(':')[0])*60 + float(x.split(':')[1]))

# Calculate mean and std dev for each row using split columns
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

# Merge the group and stroke columns from groups_df into analysis_df based on Name
analysis_df = analysis_df.merge(groups_df[['Name', 'Group', 'Stroke']], on='Name', how='left')

# Print the dataframe to verify the merge
print("\nMerged DataFrame with Group and Stroke columns:")
print(analysis_df[['Name', 'Group', 'Stroke', 'Final']].head())
print("\nFull DataFrame shape:", analysis_df.shape)


print(analysis_df.head())

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




