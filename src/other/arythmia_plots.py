import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("/arrhythmia/arrhythmia.csv")
print(df)

cor_matrix = df.corr()

upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
print(upper_tri)

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.95)]
print(to_drop)

df1 = df.drop(df.columns[to_drop], axis=1)
print(); print(df1.head())

# corr = df.corr()
#
# mask = np.zeros_like(corr, dtype=bool)
# mask[np.triu_indices_from(mask)] = True
#
# # Set background color / chart style
# sns.set_style(style='white')
#
# # Set up  matplotlib figure
# f, ax = plt.subplots(figsize=(11, 9))
#
# # Add diverging colormap from red to blue
# cmap = sns.diverging_palette(250, 10, as_cmap=True)
#
# # Draw correlation plot with or without duplicates
# sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
#             square=True,
#             linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
# plt.show()