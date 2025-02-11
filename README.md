
# Swimming Analysis Project

This project analyzes 200 free splits using the 4 50 splits, training groups and primary stroke. (subject to change)
The aim is to uncover any information that can be used to adjust training or racing strategies.

## Data Files
(Both files are omitted from the public repository, places under 'data')
- `swim_splits.csv`: Contains swimmer names and their 4 split times for 200 freestyle races, along with final times
- `group_stroke.csv`: Contains swimmer group assignments and primary strokes
- `ncaa_results/2018.csv`: Contains NCAA results for top 16 swimmers in 2018, data from 2018 to 2024 

## Analysis Scripts

### local_analysis.py
- Computes split differences and drop off rates
- Performs rank correlations
- Finds correlations between split dropoffs, variance and performance.
- Splits swimmers in to coach and stroke groups; finds correlations between split dropoffs and compares to the team wide correlations

### ncaa_analysis.py
- Computes split differences and drop off rates
- Performs rank correlations
- Finds correlations between split dropoffs, variance and performance.

### plots.py 
- Individual swimmer split progression plots
- Multi-swimmer comparison plots
- Team-wide plots

Notes:
- Function calls and prints are commented out.
- Results when ran on my data are marked with --->Results<--- inside the given function.

## Generated Plots

Plots are saved to:
- `individual_plots/`
- `team_plots/`
- `ncaa_plots/`

## Usage
Ensure data files are in the `data/` directory in the specified format and subdirectory (ncaa_results or none), or modify the scripts to read from different directories or formats.

## Results
Using data from the ASU men's swim team:

- We find there is no relationship between variance and performance
- When the data is normalized, we find there is almost no correlation (-0.08 to 0.000) between:
  - Variance vs performance
  - Drop-off vs Performance  
  - 100 Split Difference vs Performance
- In the team-wide correlation matrix: 
  - No one split difference is correlated with final time
  - Both the fade from the initial pace to the third 50 (split3_1) 
    and the fade from the initial pace to the fourth 50 (split4_1) are 
    correlated with std_dev, but the third 50 accounts for more variance .98 > .94
- Group differences can be found in group_metric_differences.csv and notable_group_correlation_differences.csv for your interpretation
  

## Ideas
- Compare NCAA top 16 performance average correlations to teams and groups
- Compare NCAA top 16 performance average correlations to individuals
- Morning vs Afternoon performance on NCAA top 16
- Track sprinter vs mid distance vs long distance
- Make group feature for coaches to select their own group 