#%matplotlib inline

# for seaborn issue:
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
#SciPy (pronounced "Sigh Pie") is a Python-based ecosystem of open-source software for mathematics, science, and engineering.
from scipy import stats
"""
scikit-learn (sklearn)
1. Simple and efficient tools for data mining and data analysis
2. Accessible to everybody, and reusable in various contexts
3. Built on NumPy, SciPy, and matplotlib
"""
import sklearn as sk
# Functions creating iterators for efficient looping
import itertools 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
"""
Seaborn is a Python visualization library based on matplotlib.
It provides a high-level interface for drawing attractive statistical graphics.
"""
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn import tree 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn import svm
"""
XGBoost is short for "Extreme Gradient Boosting".
It is used for supervised learning problems, where we use the training data (with multiple features)
"""
#import xgboost as xgb
# mlxtend : Machine Learning Library Extensions
from mlxtend.classifier import StackingClassifier

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# sns : seaborn
sns.set(style='white', context='notebook', palette='deep')

# 1. Load input data
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
combine = pd.concat([train.drop('Survived', 1), test])


# 2. Initial Exploration
"""
A broad overview
"""
print(train.head(8))
print(train.describe())

"""
To know missing values
"""
print(train.isnull().sum())
print(test.info())

surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]
surv_col = "blue"
nosurv_col = "red"

print("Survived: %i (%.1f percent), Not Survived: %i (%.1f percent), Total: %i"\
        %(len(surv), 1.*len(surv)/len(train)*100.0,\
          len(nosurv), 1.*len(nosurv)/len(train)*100.0, len(train)))

warnings.filterwarnings(action="ignore")
plt.figure(figsize=[12,10])
plt.subplot(331)
sns.distplot(surv['Age'].dropna().values, bins=range(0,81,1), kde=False, color=surv_col)
sns.distplot(nosurv['Age'].dropna().values, bins=range(0,81,1), kde=False, color=nosurv_col, axlabel='Age')
plt.subplot(332)
sns.barplot('Sex', 'Survived', data=train)
plt.subplot(333)
sns.barplot('Pclass', 'Survived', data=train)
plt.subplot(334)
sns.barplot('Embarked', 'Survived', data=train)
plt.subplot(335)
sns.barplot('SibSp', 'Survived', data=train)
plt.subplot(336)
sns.barplot('Parch', 'Survived', data=train)
plt.subplot(337)
sns.distplot(np.log10(surv['Fare'].dropna().values+1), kde=False, color=surv_col)
sns.distplot(np.log10(nosurv['Fare'].dropna().values+1), kde=False, color=nosurv_col, axlabel='Fare')
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

#print("Median age survivors: %.1f, Median age non-survivor: %.1f"\
#        %(np.median(surv['Age'].dropna()), np.median(nosurv['Age'].dropna())))

"""
#%matplotlib inline : displays the plots INSIDE the notebook
plt.show() : displays the plots OUTSIDE of the notebook
#%matplotlib inline will OVERRIDE plt.show() in the sense that plots will be shown IN the notebook even when plt.show() is called
"""
#plt.show()

tab = pd.crosstab(train['SibSp'], train['Survived'])
print(tab)

tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, color=[nosurv_col, surv_col])
plt.xlabel('SibSp')
plt.ylabel('Percentage')
#plt.show()

# binomial distribution. use binomial test to estimate the probability
print(stats.binom_test(x=5, n=5, p=0.62))

# Cabin numbers
print("We know %i of %i Cabin numbers in the training data set and" % (len(train['Cabin'].dropna()), len(train)))
print("we know %i of %i Cabin numbers in the testing data set." % (len(test['Cabin'].dropna()), len(test)))
print(train.loc[:, ['Survived', 'Cabin']].dropna().head(8))

# Ticket numbers, how many unique ticket numbers there are
print("There are %i unique ticket numbers among the %i tickets." % (train['Ticket'].nunique(), train['Ticket'].count()))

train_tkgrouped = train.groupby('Ticket')
k = 0
for name, group in train_tkgrouped:
  if (len(group) > 1):
    print(group.loc[:, ['Survived', 'Name', 'Fare']])
    k += 1
  if (k>10):
    break


# 3. Relations between features
# heatmap
plt.figure(figsize=(8,8))
foo = sns.heatmap(train.drop('PassengerId', axis=1).corr(), vmax=0.6, square=True, annot=True)
#Note: axis=1 denotes that we are referring to a column, not a row
#plt.show()

# pairplot
cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
g = sns.pairplot(data=train.dropna(), vars=cols, size=1.5, hue='Survived', palette=[nosurv_col, surv_col])
g.set(xticklabels=[])
#plt.show(g)


msurv = train[(train['Survived']==1) & (train['Sex']=='male')]
fsurv = train[(train['Survived']==1) & (train['Sex']=='female')]
mnosurv = train[(train['Survived']==0) & (train['Sex']=='male')]
fnosurv = train[(train['Survived']==0) & (train['Sex']=='female')]

plt.figure(figsize=[13,5])
plt.subplot(121)
sns.distplot(fsurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
sns.distplot(fnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col, axlabel='Female Age')
plt.subplot(122)
sns.distplot(msurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=surv_col)
sns.distplot(mnosurv['Age'].dropna().values, bins=range(0, 81, 1), kde=False, color=nosurv_col, axlabel='Male Age')

plt.close('all')
#plt.show()

#foo = combine['Age'].hist(by=combine['Pclass'], bins=np.arange(0,81,1), layout=[3,1], sharex=True, figsize=[8,12])
#foo = sns.boxplot(x="Pclass", y="Age", hue="Survived", data=train)
#sns.violinplot(x="Pclass", y="Age", data=combine, inner=None)
#sns.swarmplot(x="Pclass", y="Age", data=combine, color="w", alpha=.5)
sns.violinplot(x="Pclass", y="Age", hue="Survived", data=train, split=True)
plt.hlines([0,10], xmin=-1, xmax=3, linestyles="dotted")

dummy = mosaic(train, ["Survived", "Sex", "Pclass"])

g = sns.factorplot(x="Pclass", y="Survived", hue="Sex", col="Embarked", data=train, aspect=0.9, size=3.5, ci=95.0)
# for some reason in this plot the colours for m/f are flipped:
#grid = sns.FacetGrid(train, col='Embarked', size=2.2, aspect=1.6)
#grid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', ci=95.0, palette='deep')
#grid.add_legend()

tab = pd.crosstab(combine['Embarked'], combine['Pclass'])
print(tab)
dummp = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummp = plt.xlabel('Port embarked')
dummp = plt.ylabel('Percentage')

plt.figure()
sns.barplot(x='Embarked', y='Survived', hue='Pclass', data=train)

tab = pd.crosstab(combine['Embarked'], combine['Sex'])
print(tab)
dummpy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummpy = plt.xlabel('Port embarked')
dummpy = plt.ylabel('Percentage')

tab = pd.crosstab(combine['Pclass'], combine['Sex'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummpy = plt.xlabel('Pclass')
dummpy = plt.ylabel('Percentage')

sib = pd.crosstab(train['SibSp'], train['Sex'])
print(sib)
dummy = sib.div(sib.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Siblings')
dummy = plt.ylabel('Percentage')

parch = pd.crosstab(train['Parch'], train['Sex'])
print(parch)
dummy = parch.div(parch.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Parent/Children')
dummy = plt.ylabel('Percentage')

plt.close('all')
#plt.show()

sns.violinplot(x="Embarked", y="Age", hue="Survived", data=train, split=True)
plt.hlines([0,10], xmin=-1, xmax=3, linestyles="dotted")

plt.figure(figsize=[12,10])
plt.subplot(311)
ax1 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==1].dropna().values+1), kde=False, color=surv_col)
ax1 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==1].dropna().values+1), kde=False, color=nosurv_col, axlabel='Fare')
ax1.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplot(312)
ax2 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==2].dropna().values+1), kde=False, color=surv_col)
ax2 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==2].dropna().values+1), kde=False, color=nosurv_col, axlabel='Fare')
ax2.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplot(313)
ax3 = sns.distplot(np.log10(surv['Fare'][surv['Pclass']==3].dropna().values+1), kde=False, color=surv_col)
ax3 = sns.distplot(np.log10(nosurv['Fare'][nosurv['Pclass']==3].dropna().values+1), kde=False, color=nosurv_col, axlabel='Fare')
ax3.set_xlim(0,np.max(np.log10(train['Fare'].dropna().values)))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)

ax = sns.boxplot(x="Pclass", y="Fare", hue="Survived", data=train);
ax.set_yscale('log')

plt.close('all')
#plt.show()
    
# 4. Filling in missing values
print(train[train['Embarked'].isnull()])
print(combine.where((combine['Embarked']!='Q') & (combine['Pclass']<1.5) & (combine['Sex']=='female')).groupby(['Embarked', 'Pclass', 'Sex', 'Parch', 'SibSp']).size())

train['Embarked'].iloc[61] = "C"
train['Embarked'].iloc[829] = "C"

print(test[test['Fare'].isnull()])
test['Fare'].iloc[152] = combine['Fare'][combine['Pclass']==3].dropna().median()
print(test['Fare'].iloc[152])

# 5. Derived (engineered) features
combine = pd.concat([train.drop('Survived', 1), test])
survived = train['Survived']

combine['Child'] = combine['Age'] <= 10
combine['Cabin_known'] = combine['Cabin'].isnull() == False
combine['Age_known'] = combine['Age'].isnull() == False
combine['Family'] = combine['SibSp'] + combine['Parch']
combine['Alone'] = combine['Family'] == 0
combine['Large_Family'] = (combine['SibSp']>2) | (combine['Parch']>3)
combine['Deck'] = combine['Cabin'].str[0]
combine['Deck'] = combine['Deck'].fillna(value='U')
combine['Ttype'] = combine['Ticket'].str[0]
combine['Title'] = combine['Name'].str.split(", ", expand=True)[1].str.split(".", expand=True)[0]
combine['Fare_cat'] = pd.DataFrame(np.floor(np.log10(combine['Fare']+1))).astype('int')
combine['Bad_ticket'] = combine['Ttype'].isin(['3','4','5','6','7','8','A','L','W'])
# JOhnny, should add (combine['Age']>=10)
combine['Young'] = (combine['Age']<=30) | (combine['Title'].isin(['Master','Miss','Mlle']))
combine['Shared_ticket'] = np.where(combine.groupby('Ticket')['Name'].transform('count')>1, 1, 0)
combine['Ticket_group'] = combine.groupby('Ticket')['Name'].transform('count')
combine['Fare_eff'] = combine['Fare'] / combine['Ticket_group']
combine['Fare_eff_cat'] = np.where(combine['Fare_eff']>16.0, 2, 1)
combine['Fare_eff_cat'] = np.where(combine['Fare_eff']<8.5, 0, combine['Fare_eff_cat'])
test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train['Survived'] = survived

surv = train[train['Survived']==1]
nosurv = train[train['Survived']==0]

# Child
g = sns.factorplot(x="Sex", y="Survived", hue="Child", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
tab = pd.crosstab(train['Child'], train['Pclass'])
print(tab)
tab = pd.crosstab(train['Child'], train['Sex'])
print(tab)

#Cabin_known
cab = pd.crosstab(train['Cabin_known'], train['Survived'])
print(cab)
dummy = cab.div(cab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Cabin known')
dummy = plt.ylabel('Percentage')
g = sns.factorplot(x="Sex", y="Survived", hue="Cabin_known", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)

# Deck
tab = pd.crosstab(train['Deck'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Deck')
dummy = plt.ylabel('Percentage')
print(stats.binom_test(x=12,n=12+35, p=24/(24.+35.)))
g = sns.factorplot(x="Deck", y="Survived", hue="Sex", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
plt.close('all')
#plt.show()

# Ttype and Bad_ticket
print(train['Ttype'].unique())
print(test['Ttype'].unique())
tab = pd.crosstab(train['Ttype'], train['Survived'])
print(tab)
sns.barplot(x="Ttype", y="Survived", data=train, ci=95.0, color="blue")

tab = pd.crosstab(train['Bad_ticket'], train['Survived'])
print(tab)
g = sns.factorplot(x="Bad_ticket", y="Survived", hue="Sex", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)

tab = pd.crosstab(train['Deck'], train['Bad_ticket'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Deck')
dummy = plt.ylabel('Percentage')
plt.close('all')
#plt.show()

# Age_known
tab = pd.crosstab(train['Age_known'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Age_known')
dummy = plt.ylabel('Percentagge')

print(stats.binom_test(x=424, n=290+424, p=125/(125.+52.)))
g = sns.factorplot(x="Sex", y="Age_known", hue="Embarked", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
plt.close('all')
#plt.show()

# Family
tab = pd.crosstab(train['Family'], train['Survived'])
print(tab)
dummy = tab.div(tab.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True)
dummy = plt.xlabel('Family members')
dummy = plt.ylabel('Percentage')
plt.close('all')
#plt.show()

# Alone
tab = pd.crosstab(train['Alone'], train['Survived'])
print(tab)
sns.barplot('Alone', 'Survived', data=train)
g = sns.factorplot(x="Sex", y="Alone", hue="Embarked", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
plt.close('all')
#plt.show()

# Large_Family
tab = pd.crosstab(train['Large_Family'], train['Survived'])
print(tab)
sns.barplot('Large_Family', 'Survived', data=train)
g = sns.factorplot(x="Sex", y="Large_Family", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
plt.close('all')
#plt.show()

# Shared_ticket
tab = pd.crosstab(train['Shared_ticket'], train['Survived'])
print(tab)
sns.barplot('Shared_ticket', 'Survived', data=train)
tab = pd.crosstab(train['Shared_ticket'], train['Sex'])
print(tab)
g = sns.factorplot(x="Sex", y="Shared_ticket", hue="Embarked", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
plt.close('all')
#plt.show()

# Title
print(combine['Age'].groupby(combine['Title']).count())
print(combine['Age'].groupby(combine['Title']).mean())
print("There are %i unique titles in total." % (len(combine['Title'].unique())))

dummy = combine[combine['Title'].isin(['Mr', 'Miss', 'Mrs', 'Master'])]
foo = dummy['Age'].hist(by=dummy['Title'], bins=np.arange(0,81,1))

tab = pd.crosstab(train['Young'], train['Survived'])
print(tab)
sns.barplot('Young', 'Survived', data=train)

tab = pd.crosstab(train['Young'], train['Pclass'])
print(tab)
g = sns.factorplot(x="Sex", y="Young", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
plt.close('all')
#plt.show()

# Fare_cat
print(pd.DataFrame(np.floor(np.log10(train['Fare']+1))).astype('int').head(5))
tab = pd.crosstab(train['Fare_cat'], train['Survived'])
print(tab)
sns.barplot('Fare_cat', 'Survived', data=train)
g = sns.factorplot(x="Sex", y="Fare_cat", hue="Embarked", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
plt.close('all')
#plt.show()

# Fare_eff_cat
combine.groupby('Ticket')['Fare'].transform('std').hist()
np.sum(combine.groupby('Ticket')['Fare'].transform('std') > 0)

combine.iloc[np.where(combine.groupby('Ticket')['Fare'].transform('std') > 0)]

plt.figure(figsize=[12,10])
plt.subplot(311)
ax1 = sns.distplot(np.log10(surv['Fare_eff'][surv['Pclass']==1].dropna().values+1), kde=False, color=surv_col)
ax1 = sns.distplot(np.log10(nosurv['Fare_eff'][nosurv['Pclass']==1].dropna().values+1), kde=False, color=nosurv_col, axlabel='Fare')
ax1.set_xlim(0,np.max(np.log10(train['Fare_eff'].dropna().values)))
plt.subplot(312)
ax2 = sns.distplot(np.log10(surv['Fare_eff'][surv['Pclass']==2].dropna().values+1), kde=False, color=surv_col)
ax2 = sns.distplot(np.log10(nosurv['Fare_eff'][nosurv['Pclass']==2].dropna().values+1), kde=False, color=nosurv_col, axlabel='Fare')
ax2.set_xlim(0,np.max(np.log10(train['Fare_eff'].dropna().values)))
plt.subplot(313)
ax3 = sns.distplot(np.log10(surv['Fare_eff'][surv['Pclass']==3].dropna().values+1), kde=False, color=surv_col)
ax3 = sns.distplot(np.log10(nosurv['Fare_eff'][nosurv['Pclass']==3].dropna().values+1), kde=False, color=nosurv_col, axlabel='Fare')
ax3.set_xlim(0,np.max(np.log10(train['Fare_eff'].dropna().values)))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25, wspace=0.35)
plt.close('all')
#plt.show()

print(combine[combine['Fare']>1].groupby('Pclass')['Fare'].std())
print(combine[combine['Fare_eff']>1].groupby('Pclass')['Fare_eff'].std())
print(combine[(combine['Pclass']==1) & (combine['Fare_eff']>0) & (combine['Fare_eff']<10)])
print(combine[(combine['Pclass']==3) & (np.log10(combine['Fare_eff'])>1.2)])

ax = sns.boxplot(x="Pclass", y="Fare_eff", hue="Survived", data=train)
ax.set_yscale('log')
ax.hlines([8.5,16], -1, 4, linestyles='dashed')

plt.figure()
tab = pd.crosstab(train['Fare_eff_cat'], train['Survived'])
print(tab)
sns.barplot('Fare_eff_cat', 'Survived', data=train)
g = sns.factorplot(x="Sex", y="Fare_eff_cat", hue="Embarked", col="Pclass", data=train, aspect=0.9, size=3.5, ci=95.0)
plt.close('all')
#plt.show()


''' combine = pd.concat([train.drop('Survived',1), test])
survived = train['Survived']

combine["Sex"] = combine["Sex"].astype("category")
combine["Sex"].cat.categories = [0,1]
combine["Sex"] = combine["Sex"].astype("int")
combine["Embarked"] = combine["Embarked"].astype("category")
combine["Embarked"].cat.categories = [0,1,2]
combine["Embarked"] = combine["Embarked"].astype("int")
combine["Deck"] = combine["Deck"].astype("category")
combine["Deck"].cat.categories = [0,1,2,3,4,5,6,7,8]
combine["Deck"] = combine["Deck"].astype("int")

test = combine.iloc[len(train):]
train = combine.iloc[:len(train)]
train['Survived'] = survived

print(train.loc[:, ["Sex", "Embarked"]].head()) '''