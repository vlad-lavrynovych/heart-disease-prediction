# import warnings filter
from warnings import simplefilter
from sklearn import metrics
from sklearn.metrics import roc_curve

import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 330)

np.set_printoptions(linewidth=330)

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
# import warnings filter
from warnings import simplefilter

# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
res = []
df = pd.read_csv('C:\\Users\\dell\\Desktop\\heart-disease-prediction\\data\\atherosclerosis\\data.csv')
print("\n\n\n\n\n\n\n")
print(df)
print(df[df.Progress == 1].age.mean())
print(df[df.Progress == 0].age.mean())
print("\n\n\n\n\n\n\n")
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
plt.ylabel('mean(age)')

plt.show()

df['sex'] = df['sex'].map({'female': 1, 'male': 0})
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
ax.set_ylabel("Cholesterin", labelpad=30, fontsize=18)
ax.set_zlabel(
    "Age",
    labelpad=10, fontsize=18)

ax.legend(prop={'size': 30})

plt.show()



df = pd.read_csv('C:\\Users\\dell\\Desktop\\heart-disease-prediction\\data\\atherosclerosis\\data.csv')
import numpy
################################## data preprocessing
X = df.iloc[:, 0:12].values
y = df.iloc[:, 13].values
print("1 ===== ")
print(len(df.iloc[:, 1:]))
print("2 ===== ")
print(len(df.iloc[:, 0]))
models = []
from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=251)

from sklearn.preprocessing import StandardScaler as ss

sc = ss()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#########################################   SVM   #############################################################
from sklearn.svm import SVC

classifier = SVC(kernel='rbf')
history = classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
models.append(classifier)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
print(y_pred_train)
cm_train = confusion_matrix(y_pred_train, y_train)

res.append(metrics.classification_report(y_train, y_pred_train))

fpr1, tpr1, thresh1 = roc_curve(y_test, y_pred, pos_label=1)


print()
print('Accuracy for training set for svm = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for svm = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot()
plt.show()

#########################################   Naive Bayes  #############################################################

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=251)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
models.append(classifier)
from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

res.append(metrics.classification_report(y_train, y_pred_train))
fpr2, tpr2, thresh2 = roc_curve(y_test, y_pred, pos_label=1)

print()
print('Accuracy for training set for Naive Bayes = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Naive Bayes = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))


disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot()
plt.show()

#########################################   Logistic Regression  #############################################################
# X = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values
#
# from sklearn.model_selection import train_test_split
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=251)
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

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=251)

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

classifier = DecisionTreeClassifier()
classifier.fit(X_train, y_train)
print(df.columns)
dotfile = open("s.png", 'w')
tree.export_graphviz(classifier, out_file=dotfile)
dotfile.close()
# Predicting the Test set results
y_pred = classifier.predict(X_test)
models.append(classifier)
from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)

res.append(metrics.classification_report(y_train, y_pred_train))
fpr3, tpr3, thresh3 = roc_curve(y_test, y_pred, pos_label=1)

print()
print('Accuracy for training set for Decision Tree = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Decision Tree = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot()
plt.show()

plt.figure(figsize=(16,19))
tree.plot_tree(classifier, feature_names=df.iloc[:, 1:].columns.values.tolist())
plt.show()

#########################################  Random Forest  #############################################################


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=251)

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=10)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
models.append(classifier)
from sklearn.metrics import confusion_matrix

cm_test = confusion_matrix(y_pred, y_test)

y_pred_train = classifier.predict(X_train)
cm_train = confusion_matrix(y_pred_train, y_train)
res.append(metrics.classification_report(y_train, y_pred_train))
fpr4, tpr4, thresh4 = roc_curve(y_test, y_pred, pos_label=1)
print()
print('Accuracy for training set for Random Forest = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set for Random Forest = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot()
plt.show()

###############################################################################
# applying lightGBM
# import lightgbm as lgb
#
# d_train = lgb.Dataset(X_train, label=y_train)
# params = {}
#
# clf = lgb.train(params, d_train, 100)
# # Prediction
# y_pred = clf.predict(X_test)
# # convert into binary values
# for i in range(0, len(y_pred)):
#     if y_pred[i] >= 0.5:  # setting threshold to .5
#         y_pred[i] = 1
#     else:
#         y_pred[i] = 0
#
# from sklearn.metrics import confusion_matrix
#
# cm_test = confusion_matrix(y_pred, y_test)
#
# y_pred_train = clf.predict(X_train)
#
# for i in range(0, len(y_pred_train)):
#     if y_pred_train[i] >= 0.5:  # setting threshold to .5
#         y_pred_train[i] = 1
#     else:
#         y_pred_train[i] = 0
#
# cm_train = confusion_matrix(y_pred_train, y_train)
# print()
# print('Accuracy for training set for LightGBM = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
# print('Accuracy for test set for LightGBM = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))

###############################################################################
# applying XGBoost

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, target, test_size = 0.20, random_state = 0)

from xgboost import XGBClassifier

xg = XGBClassifier(use_label_encoder=False)
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
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot()
plt.show()


from tensorflow.keras.models import Sequential
from keras.layers import Dense, Input, Dropout, Flatten

model = Sequential()
model.add(Dense(13, input_dim=12, input_shape=(12,)))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(13, activation='relu'))
model.add(Dense(13, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#
#
# # define two sets of inputs
# inputW = Input(shape=(len(ref.input_c1_list),))
# inputX = Input(shape=(len(ref.input_c2_list),))
# inputY = Input(shape=(len(ref.input_n1_list),))
#
# # the first branch operates on the first input
# w = Dense(3, activation="relu", input_dim=len(ref.input_c1_list))(inputW)
# #w = Dense(1, activation="relu")(w)
# w = Model(inputs=inputW, outputs=w)
#
# # the second branch operates on the first input
# x = Dense(1, activation="relu", input_dim=len(ref.input_c2_list))(inputX)
# # x = Dense(2, activation="relu")(x)
# x = Model(inputs=inputX, outputs=x)
#
# # the third branch operates on the third input
# y = Dense(5, activation="relu", input_dim=len(ref.input_n1_list))(inputY)
# y = Dense(2, activation="relu")(y)
# #y = Dense(1, activation="relu")(y)
# y = Model(inputs=inputY, outputs=y)
#
# # combine the output of the two branches
# combined = concatenate([w.output, x.input, y.output])
#
# # apply a FC layer and then a regression prediction on the
# # combined outputs
# z = Dense(3, activation="relu")(combined)
# z = Dense(1, activation="sigmoid")(z)
#
# # our model will accept the inputs of the two branches and
# # then output a single value
# model = Model(inputs=[w.input, x.input, y.input], outputs=z)

# # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())



history = model.fit(X_train, y_train, batch_size=100, validation_split=0.2, epochs=500)
print(model.summary())
models.append(model)
y_pred = [round(float(x)) for x in model.predict(X_test)]
y_pred_train = [round(float(x)) for x in model.predict(X_train)]
fpr5, tpr5, thresh5 = roc_curve(y_test, y_pred, pos_label=1)

cm_test = confusion_matrix(y_pred, y_test)

cm_train = confusion_matrix(y_pred_train, y_train)
res.append(metrics.classification_report(y_train, y_pred_train))
print()
print('Accuracy for training set for Neural network = {}'.format((cm_train[0][0] + cm_train[1][1]) / len(y_train)))
print('Accuracy for test set forNeural network = {}'.format((cm_test[0][0] + cm_test[1][1]) / len(y_test)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_test)
disp.plot()
plt.show()

print("============================AASD")
print(X_train[0])
print("============================1 - CRIT TEST")
print(model.predict([[1, 0, 55, 175, 100,  24.7, 1, 90,  153, 93, 1,  3, 0]]))
print(classifier.predict([[1, 0, 55, 175, 100,  23.68, 1, 90,  153, 93, 1,  3, 0]]))
print("============================1 - NORM TEST")
print(model.predict([[0, 0, 20, 175, 70,  26.68, 1, 65,  120, 80, 0,  0, 0]]))
print(classifier.predict([[0, 0, 20, 175, 70,  26.68, 1, 65,  120, 80, 0,  0, 0]]))
print("============================2 50/50")
print(model.predict([[0, 0, 25, 175, 75,  23.42, 1, 60,  120, 80, 0,  2, 0]]))
print(classifier.predict([[0, 0, 25, 175, 80,  23.42, 1, 60,  120, 80, 0,  2, 0]]))
print("============================3")


res.append(metrics.classification_report(y_test, y_pred))

from keras.utils.vis_utils import plot_model

plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

from matplotlib import pyplot as plt

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('modell loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

print(len(res))

print("\n==============1SVC", res[0])
print("\n==============2NAIVE BAYES", res[1])
print("\n==============3DTREE", res[2])
print("\n==============4RANDOMFOREST", res[3])
print("\n==============5NEURAL NETWORK", res[4])

for i in res:
    print(i)

import matplotlib.pyplot as plt
plt.style.use('seaborn')

random_probs = [0 for i in range(len(y_test))]
p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)

# plot roc curves
plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='SVC')
plt.plot(fpr2, tpr2, linestyle='--',color='green', label='Naive Bayes')
plt.plot(fpr3, tpr3, linestyle='--',color='yellow', label='Decision Tree')
plt.plot(fpr4, tpr4, linestyle='--',color='purple', label='Random Forest')
plt.plot(fpr5, tpr5, linestyle='--',color='black', label='Neural Network')
plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')

# title
plt.title('ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('ROC',dpi=300)
plt.show()

from ann_visualizer.visualize import ann_viz

# ann_viz(model, title="", view=True, filename="model.gv")
