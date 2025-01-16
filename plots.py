import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# Read the CSV
df = pd.read_csv('data/swim_splits.csv')

def plot_swimmer_splits(name, df):
    """
    Creates a line plot of an individual swimmer's splits across their 200 Free.
    
    Args:
        name (str): The swimmer's name to plot
        df (pandas.DataFrame): DataFrame containing swim split data with columns 
            [Name, Split1, Split2, Split3, Split4]
    """
    swimmer_data = df[df['Name'] == name]
    
    splits = swimmer_data[['Split1', 'Split2', 'Split3', 'Split4']].values[0]
    
    x_points = [1,2,3,4]
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_points, splits, 'bo-', linewidth=2, markersize=8)
    
    plt.title(f'200 Free')
    plt.xlabel('Split')
    plt.ylabel('Split Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(x_points)

    plt.ylim(min(splits) - 1, max(splits) + 1)
    
    name = name.replace(" ", "_", 1)
    plt.savefig(f'individual_plots/{name}_splits.png')

#for i in df['Name']:
#    plot_swimmer_splits(i, df)

def plot_multiple_swimmers(names, df):
    """
    Creates a comparative line plot of multiple swimmers' splits.
    
    Args:
        names (list): List of swimmer names to compare
        df (pandas.DataFrame): DataFrame containing swim split data with columns 
            [Name, Split1, Split2, Split3, Split4]
    """
    x_points = [1, 2, 3, 4]
    
    plt.figure(figsize=(12, 8))
    
    all_splits = []

    for name in names:
        swimmer_data = df[df['Name'] == name]
        
        splits = swimmer_data[['Split1', 'Split2', 'Split3', 'Split4']].values[0]
        all_splits.extend(splits)
        
        plt.plot(x_points, splits, 'o-', linewidth=2, markersize=8, label=name)
    
    plt.title('Split Comparison')
    plt.xlabel('Split Number')
    plt.ylabel('Time (seconds)')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.xticks(x_points)
    
    plt.ylim(min(all_splits) - 1, max(all_splits) + 1)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()  
    plt.show()

#swimmers_to_plot = ['Jack Dolan', 'Tiago Behar', 'Jonny Kulow', 'Quin Seider', 'Filip Senc-Samardzic']
#plot_multiple_swimmers(swimmers_to_plot, df)

def plot_splits_with_mean(df):
    """
    Creates a scatter plot showing each swimmer's four splits and their mean.
    
    Args:
        df (pandas.DataFrame): DataFrame containing swim split data with columns 
            [Name, Split1, Split2, Split3, Split4]
    """
    plt.figure(figsize=(12, 6))
    
    # Sort swimmers by mean time
    df['Mean'] = df[['Split1', 'Split2', 'Split3', 'Split4']].mean(axis=1)
    df_sorted = df.sort_values('Mean', ascending=True)
    
    for idx, (_, swimmer) in enumerate(df_sorted.iterrows()):
        # Plot individual splits
        splits = [swimmer['Split1'], swimmer['Split2'], 
                 swimmer['Split3'], swimmer['Split4']]
        plt.scatter([idx] * 4, splits, color='darkblue', alpha=1.0, s=30)
        
        # Plot mean with different marker
        plt.scatter(idx, swimmer['Mean'], color='darkred', marker='^', 
                   s=50, label='Mean' if idx == 0 else "")
    
    plt.xticks(range(len(df_sorted)), df_sorted['Name'], rotation=45, ha='right')
    plt.ylabel('Time (seconds)')
    plt.title('Split Times by Swimmer (with Mean)')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis ticks to increments of 0.5
    ymin = math.floor(df_sorted[['Split1', 'Split2', 'Split3', 'Split4']].min().min() * 2) / 2
    ymax = math.ceil(df_sorted[['Split1', 'Split2', 'Split3', 'Split4']].max().max() * 2) / 2
    plt.yticks(np.arange(ymin, ymax + 0.5, 0.5))
    
    # Add legend
    plt.scatter([], [], color='darkblue', label='Individual Splits')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('team_plots/splits.png')
    plt.close()

plot_splits_with_mean(df)