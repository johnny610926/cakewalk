import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
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
from sklearn.metrics import confusion_matrix

#import xgboost as xgb

train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")
combine = pd.concat([train.drop('Survived',1), test])

# 4. Filling in missing values
train['Embarked'].iloc[61] = "C"
train['Embarked'].iloc[829] = "C"
test['Fare'].iloc[152] = combine['Fare'][combine['Pclass']==3].dropna().median()

# 5. Derived (engineered) features
combine = pd.concat([train.drop('Survived',1), test])
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

# 6. Preparing for modelling
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

# Splitting the train sample into two sub-samples: training and testing
training, testing = train_test_split(train, test_size=0.2, random_state=0)
print("Total sample size = %i; training sample size = %i, testing sample size = %i"\
        %(train.shape[0], training.shape[0], testing.shape[0]))

# Test and select the model features
cols = ['Sex', 'Pclass', 'Cabin_known', 'Large_Family', 'Shared_ticket', 'Young', 'Alone', 'Child']
tcols = np.append(['Survived'], cols)

df = training.loc[:, tcols].dropna()
X = df.loc[:, cols]
y = np.ravel(df.loc[:,['Survived']])

df_test = testing.loc[:, tcols].dropna()
X_test = df_test.loc[:, cols]
y_test = np.ravel(df_test.loc[:,['Survived']])
''' 
clf_ext = ExtraTreesClassifier(max_features='auto', bootstrap=True, oob_score=True)
param_grid = {  "criterion" : ["gini", "entropy"],
                "min_samples_leaf" : [1, 5, 10],
                "min_samples_split" : [8, 10, 12],
                "n_estimators" : [20, 50, 100]}
gs = GridSearchCV(estimator=clf_ext, param_grid=param_grid, scoring='accuracy', cv=3)
gs = gs.fit(X,y)
print(gs.best_score_)
print(gs.best_params_)
 '''
clf_ext = ExtraTreesClassifier(
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    criterion='gini',
    min_samples_leaf=5,
    min_samples_split=8,
    n_estimators=50
)
clf_ext = clf_ext.fit(X,y)
score = clf_ext.score(X,y)
print("Score of training set = ", score)
print(pd.DataFrame(list(zip(X.columns, np.transpose(clf_ext.feature_importances_))), columns=['Feature','Importance']).sort_values(by='Importance', ascending=False))

''' 
import taner_code
class_names = ["Dead", "Alive"]
cnf_matrix = confusion_matrix(clf_ext.predict(X_test),y_test)
taner_code.show_confusion_matrix(cnf_matrix,class_names)
plt.show()
'''

# Model validation
''' 
clf = clf_ext
scores = cross_val_score(clf, X, y, cv=5)
print(scores)
print("Mean score = %.3f, Std deviation = %.3f" % (np.mean(scores), np.std(scores)))
'''
score_ext_test = clf_ext.score(X_test, y_test)
print("Score of corss-validation set =", score_ext_test)
