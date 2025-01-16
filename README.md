
# Swimming Analysis Project

This project analyzes 200 free splits using the 4 50 splits, training groups and primary stroke. (subject to change)
The aim is to uncover any information that can be used to adjust training or racing strategies.

## Data Files
(Both files are omitted from the public repository)
- `swim_splits.csv`: Contains swimmer names and their 4 split times for 200 freestyle races, along with final times
- `group_stroke.csv`: Contains swimmer group assignments and primary strokes

## Analysis Scripts

### analysis.py
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

## Usage
Ensure data files are in the `data/` directory in the specified format, or modify the scripts to read from different directories or formats.

## Findings
Using data from the ASU men's swim team:

- We find there is no relationship between variance and performance
- When the data is normalized, we find there is almost no correlation (-0.08 to 0.000) between:
  - Variance vs performance
  - Drop-off vs Performance  
  - 100 Split Difference vs Performance
- Correlation matrix analysis is in progress

