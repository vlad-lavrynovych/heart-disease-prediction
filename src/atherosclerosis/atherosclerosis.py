# import warnings filter
from warnings import simplefilter

import numpy as np
import pandas as pd

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

df = pd.read_csv('C:\\Users\\dell\\Desktop\\heart-disease-prediction\\data\\atherosclerosis\\data.csv')
# Progres,OP,Shunt,vik,zrist,vaha,IMT,stat,ChSS,AD sist.,AD diast,AG therapia,cholesterin,diabetus melitus
### 1 = male, 0 = female
print(df.isnull().sum())  # -- no missing values

df['sex'] = df.sex.map({0: 'male', 1: 'female'})

import matplotlib.pyplot as plt

import seaborn as sns

# distribution of target vs age
sns.set_context(font_scale=2, rc={"font.size": 20, "axes.titlesize": 25, "axes.labelsize": 20})

s = sns.catplot(height=13, aspect=2.1, kind='count', data=df, x='age', hue='Progress',
                order=df['age'].sort_values().unique())
s.legend.set_title('Atherosclerosis')
plt.title('Variation of Age for each target class')
plt.show()

# barplot of age vs sex with hue = target
g = sns.catplot(height=13, aspect=1.2, kind='bar', data=df, y='age', x='sex', hue='Progress', legend_out=False)
g.legend.set_title('Atherosclerosis')
plt.title('Distribution of age vs sex with the target class')

plt.show()

corr = df.corr()
mask = np.zeros_like(corr, dtype=bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(250, 10, as_cmap=True)
f, ax = plt.subplots(figsize=(11, 9))
# Draw correlation plot with or without duplicates
sns.heatmap(corr, cmap=cmap, vmin=-1, vmax=1,
            square=True,
            linewidth=.5, cbar_kws={"shrink": .5}, ax=ax)
plt.show()

fig = plt.figure(figsize=(20, 16))

ax = fig.add_subplot(111, projection='3d')

h = df[df.Progress == 1]
i = df[df.Progress == 0]

print(h)

ax.scatter(h.weight, h.cholesterin, h.age, marker="o", c="red", label='Sick', s=100)
ax.scatter(i.weight, i.cholesterin, i.age, marker="o", c="green", label='Healthy', s=100)

ax.set_title("Weight, Age and Cholesterin distribution by target classes", fontsize=40)

ax.set_xlabel(
    "Weight",
    labelpad=25, fontsize=18)
ax.set_ylabel("Cholesterin",labelpad=30, fontsize=18)
ax.set_zlabel(
    "Age",
    labelpad=10, fontsize=18)

ax.legend(prop={'size': 30})

plt.show()

df['sex'] = df['sex'].map({'female': 1, 'male': 0})

################################## data preprocessing
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler as ss

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#########################################   SVM   #############################################################
from sklearn.svm import SVC

classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
print(y_pred_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

#########################################   Naive Bayes  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

#########################################   Logistic Regression  #############################################################
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#
# from sklearn.linear_model import LogisticRegression
#
# classifier = LogisticRegression()
# classifier.fit(X_train, y_train)
#
# # Predicting the Test set results
# y_pred = classifier.predict(X_test)
#
# from sklearn.metrics import confusion_matrix
#
# cm_test = confusion_matrix(y_pred, y_test)
#
# y_pred_train = classifier.predict(X_train)
# cm_train = confusion_matrix(y_pred_train, y_train)
#
# print()
# print('Accuracy for training set for Logistic Regression = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
# print('Accuracy for test set for Logistic Regression = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

#########################################   Decision Tree  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

#########################################  Random Forest  #############################################################
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

print()
print('Accuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

###############################################################################
# applying lightGBM
import lightgbm as lgb

d_train = lgb.Dataset(X_train, label=y_train)
params = {}

clf = lgb.train(params, d_train, 100)
# Prediction
y_pred = clf.predict(X_test)
# convert into binary values
for i in range(0, len(y_pred)):
    if y_pred[i] >= 0.5:  # setting threshold to .5
        y_pred[i] = 1
    else:
        y_pred[i] = 0

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = clf.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i] >= 0.5:  # setting threshold to .5
        y_pred_train[i] = 1
    else:
        y_pred_train[i] = 0

cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for LightGBM = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for LightGBM = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

###############################################################################
# applying XGBoost

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.20, random_state = 0)

from xgboost import XGBClassifier

xg = XGBClassifier()
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = xg.predict(X_train)

for i in range(0, len(y_pred_train)):
    if y_pred_train[i] >= 0.5:  # setting threshold to .5
        y_pred_train[i] = 1
    else:
        y_pred_train[i] = 0

cm_train = confusion_matrix(y_pred_train, y_train)
print()
print('Accuracy for training set for XGBoost = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for XGBoost = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
