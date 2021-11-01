import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("C:\\Users\\dell\\Desktop\\heart-disease-prediction\\data\\uci\\heart.csv")
print(df)

# Fixing random state for reproducibility
np.random.seed(19680801)

plt.rcdefaults()
fig, ax = plt.subplots()

labels = ('Male', 'Female')
y_pos = np.arange(2)
present = df[df.sex == 1].sex.count()
notPresent = df[df.sex == 0].sex.count()

performance = 3 + 10 * np.random.rand(2)
error = np.random.rand(2)

ax.barh(y_pos, [present, notPresent], xerr=error, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of patients')
ax.set_title('Gender distribution')

plt.show()
# ==============================
plt.rcdefaults()
fig, ax = plt.subplots()

labels = ('Present', 'Not present')
y_pos = np.arange(2)
present = df[df.target == 1].target.count()
notPresent = df[df.target == 0].target.count()

performance = 3 + 10 * np.random.rand(2)
error = np.random.rand(2)

ax.barh(y_pos, [present, notPresent], xerr=error, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Number of patients')
ax.set_title('Disease distribution')

plt.show()
# =====================================
plt.style.use('ggplot')
ages = df[df.target == 1].age
print(ages)
plt.hist(ages)
plt.xlabel("Age")
plt.ylabel("Number of patients")
plt.title("Age Distribution Among Patients with a Disease")
plt.show()
# =====================================

corr = df.corr()

mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

# Set background color / chart style
sns.set_style(style='white')

# Set up  matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Add diverging colormap from red to blue
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Draw correlation plot with or without duplicates
sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
            square=True,
            linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()
# =============================================
fig = plt.figure(figsize=(20, 16))

ax = fig.add_subplot(111, projection='3d')


ax.scatter(df[df.target == 1].cp, df[df.target == 1].thalach, df[df.target == 1].slope, marker="o", c="red", label='Sick', s=100)
ax.scatter(df[df.target == 0].cp, df[df.target == 0].thalach, df[df.target == 0].slope, marker="o", c="green", label='Healthy', s=100)


ax.set_title("Correlated parameters distribution", fontsize=40)

ax.set_xlabel("Chest pain " + "\n-- Value 1: typical angina\n-- Value 2: atypical angina\n-- Value 3: non-anginal pain\n-- Value 4: asymptomatic", labelpad=25, fontsize=18)
ax.set_ylabel("Maximum heart rate achieved", fontsize=18)
ax.set_zlabel("The slope of the peak exercise ST segment\n-- Value 1: upsloping\n-- Value 2: flat\n-- Value 3: downsloping", labelpad=30, fontsize=18)

ax.set_xlim3d(0, 4)
ax.set_ylim3d(0, 200)
ax.set_zlim3d(0, 3)

ax.legend(prop={'size': 30})

plt.show()
# =============================================
