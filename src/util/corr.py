import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# A list with your data slightly edited
l = [1.0, 0.00279981, 0.95173379, 0.02486161, -0.00324926, -0.00432099,
     0.00279981, 1.0, 0.17728303, 0.64425774, 0.30735071, 0.37379443,
     0.95173379, 0.17728303, 1.0, 0.27072266, 0.02549031, 0.03324756,
     0.02486161, 0.64425774, 0.27072266, 1.0, 0.18336236, 0.18913512,
     -0.00324926, 0.30735071, 0.02549031, 0.18336236, 1.0, 0.77678274,
     -0.00432099, 0.37379443, 0.03324756, 0.18913512, 0.77678274, 1.00]

# Split list
n = 6
data = [l[i:i + n] for i in range(0, len(l), n)]

# A dataframe
df = pd.DataFrame(data)


def CorrMtx(df, dropDuplicates=True):
    # Your dataset is already a correlation matrix.
    # If you have a dateset where you need to include the calculation
    # of a correlation matrix, just uncomment the line below:
    # df = df.corr()

    # Exclude duplicate correlations by masking uper right values
    if dropDuplicates:
        mask = np.zeros_like(df, dtype=np.bool)
        mask[np.triu_indices_from(mask)] = True

    # Set background color / chart style
    sns.set_style(style='white')

    # Set up  matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Add diverging colormap from red to blue
    cmap = sns.diverging_palette(250, 10, as_cmap=True)

    # Draw correlation plot with or without duplicates
    if dropDuplicates:
        sns.heatmap(df, mask=mask, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
    else:
        sns.heatmap(df, cmap=cmap,
                    square=True,
                    linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)


CorrMtx(df, dropDuplicates=False)
