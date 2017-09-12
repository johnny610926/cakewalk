import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
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

import xgboost as xgb

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
#print(train.loc[:, ["Sex", "Embarked"]].head())
''' 
ax = plt.subplots(figsize=(12,10))
foo = sns.heatmap(train.drop('PassengerId', axis=1).corr(), vmax=1.0, square=True, annot=True)
plt.show()
'''

# 7. Modelling
'''
Let's summarise briefly what we found in our data exploration
- sex and ticket class are the main factors
- there seem to be additional impacts from:
    > age: young men vs young women; (male) children
    > relatives: parch=1-3, sibsp=1-2(somewhat explained by sex but no completely)
    > maybe the cabin deck, but not many are known
- other apparent effects appear to be strongly connected to the sex/class features:
    > port of embarkation
    > fare
    > sharing a ticket
    > large family
    > travelling alone
    > known cabin number
    > known age
'''

# Splitting the train sample into two sub-samples: training and testing
training, testing = train_test_split(train, test_size=0.2, random_state=0)
print("Total sample size = %i; training sample size = %i, testing sample size = %i"\
        %(train.shape[0], training.shape[0], testing.shape[0]))

# Test and select the model features
cols = ['Sex', 'Pclass', 'Cabin_known', 'Large_Family', 'Parch', 'SibSp', 'Young', 'Alone', 'Shared_ticket', 'Child']
tcols = np.append(['Survived'], cols)

df = training.loc[:, tcols].dropna()
X = df.loc[:,cols]
y = np.ravel(df.loc[:,['Survived']])
clf_log = LogisticRegression()
clf_log = clf_log.fit(X,y)
score_log = clf_log.score(X,y)
print("------------ Logistic regression(First time) ------------")
print(cols)
print(score_log)
print(pd.DataFrame(list(zip(X.columns, np.transpose(clf_log.coef_))), columns=['Feature', 'Importance']))
print()

clf_score = {} # Record classifier's performance score

# Run and describe several different classifiers
cols = ['Sex', 'Pclass', 'Cabin_known', 'Large_Family', 'Shared_ticket', 'Young', 'Alone', 'Child']
tcols = np.append(['Survived'], cols)

df = training.loc[:, tcols].dropna()
X = df.loc[:, cols]
y = np.ravel(df.loc[:,['Survived']])

df_test = testing.loc[:, tcols].dropna()
X_test = df_test.loc[:, cols]
y_test = np.ravel(df_test.loc[:,['Survived']])

print("Selected features:", cols)
# Logistic regression
clf_log = LogisticRegression()
clf_log = clf_log.fit(X, y)
score = cross_val_score(clf_log, X, y, cv=5).mean()
#print("Logistic regression - score = %f" % score)
clf_score['Logistic Regression'] = score

# Perceptron
clf_pctr = Perceptron(class_weight='balanced')
clf_pctr = clf_pctr.fit(X, y)
score = cross_val_score(clf_pctr, X, y, cv=5).mean()
#print("Perceptron - score = %f" % score)
clf_score['Perceptron'] = score

# K Nearest Neighbours
clf_knn = KNeighborsClassifier(n_neighbors=10, weights='distance')
clf_knn = clf_knn.fit(X, y)
score = cross_val_score(clf_knn, X, y, cv=5).mean()
#print("KNN - score = %f" % score)
clf_score['KNN'] = score

# Support Vector Machines
clf_svm = svm.SVC(class_weight='balanced')
clf_svm.fit(X, y)
score = cross_val_score(clf_svm, X, y, cv=5).mean()
#print("SVM - score = %f" % score)
clf_score['Support Vector Machines'] = score

# Bagging
bagging = BaggingClassifier(KNeighborsClassifier(n_neighbors=2, weights='distance'), oob_score=True, max_samples=0.5, max_features=1.0)
clf_bag = bagging.fit(X,y)
score = clf_bag.oob_score_
#print("Bagging - score = %f" % score)
clf_score['Bagging KNN'] = score

# Decision Tree
clf_tree = tree.DecisionTreeClassifier(
    #max_depth=3,\
    class_weight="balanced",\
    min_weight_fraction_leaf=0.01\
)
clf_tree = clf_tree.fit(X,y)
score = cross_val_score(clf_tree, X, y, cv=5).mean()
#print("Decision Tree - score = %f" % score)
clf_score['Decision Tree'] = score

# Random Forest
clf_rf = RandomForestClassifier(
    n_estimators=1000,\
    max_depth=None,\
    min_samples_split=10\
    #class_weight="balanced",\
    #min_weight_fraction_leaf=0.02\
)
clf_rf = clf_rf.fit(X,y)
score = cross_val_score(clf_rf, X, y, cv=5).mean()
#print("Random Forest - score = %f" % score)
clf_score['Random Forest'] = score

# Extremely Randomised Trees
clf_ext = ExtraTreesClassifier(
    max_features='auto',
    bootstrap=True,
    oob_score=True,
    n_estimators=1000,
    max_depth=None,
    min_samples_split=10
    #class_weight="balanced",
    #min_weight_fraction_leaf=0.02
)
clf_ext = clf_ext.fit(X,y)
score = cross_val_score(clf_ext, X, y, cv=5).mean()
#print("Extremely Randomised Trees - score = %f" % score)
clf_score['ExtraTree'] = score

# Gradient Boosting
warnings.filterwarnings("ignore")

clf_gb = GradientBoostingClassifier(
    #loss='exponential',
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=3,
    subsample=0.5,
    random_state=0
)
clf_gb.fit(X,y)
score = cross_val_score(clf_gb, X, y, cv=5).mean()
#print("Gradient Boosting - score = %f" % score)
clf_score['Gradient Boosting'] = score

# Ada Boost
clf_ada = AdaBoostClassifier(n_estimators=400, learning_rate=0.1)
clf_ada.fit(X,y)
score = cross_val_score(clf_ada, X, y, cv=5).mean()
#print("Ada Boost - score = %f" % score)
clf_score['Ada Boost'] = score

# eXtreme Gradient Boosting
clf_xgb = xgb.XGBClassifier(
    max_depth=2,
    n_estimators=500,
    subsample=0.5,
    learning_rate=0.1
)
clf_xgb.fit(X,y)
score = cross_val_score(clf_xgb, X, y, cv=5).mean()
#print("eXtreme Gradient Boosting - score = %f" % score)
clf_score['XGBoost'] = score


#print(pd.DataFrame(list(zip(clf_score.keys(), clf_score.values())), columns=['Model','Score']))
#print(pd.DataFrame(list(clf_score.items())))
#print(pd.DataFrame(sorted(clf_score.items(), key=lambda x: x[1], reverse=True), columns=['Model','Score']))
#print(pd.DataFrame(list(clf_score.items())).sort_values(1, ascending=False))
print(pd.DataFrame(list(clf_score.items()), columns=['Model','Score']).sort_values(by='Score', ascending=False))

summary = pd.DataFrame(list(zip(X.columns,\
    np.transpose(clf_log.coef_),\
    np.transpose(clf_pctr.coef_),\
    ['U']*len(X.columns),\
    np.transpose(clf_svm._get_coef()),\
    ['U']*len(X.columns),\
    np.transpose(clf_tree.feature_importances_),\
    np.transpose(clf_rf.feature_importances_),\
    np.transpose(clf_ext.feature_importances_),\
    np.transpose(clf_gb.feature_importances_),\
    np.transpose(clf_ada.feature_importances_),\
    np.transpose(clf_xgb.feature_importances_)    
    )), columns=['Feature','Logistic', 'Perceptron', 'KNN', 'SVM', 'Bagging', 'Tree', 'RF', 'Extra', 'GB', 'Ada', 'Xtreme'])
summary['Median'] = summary.median(1)
summary.sort_values('Median', ascending=False)
print(summary)

# Stacking / Ensemble methods
clf_vote = VotingClassifier(
    estimators=[
        #('tree', clf_tree)
        ('knn', clf_knn),
        ('svm', clf_svm),
        ('extra', clf_ext),
        #('gb', clf_gb),
        ('xgb', clf_xgb),
        ('percep', clf_pctr),
        ('logistic', clf_log),
        #('RF', clf_rf)
    ],
    weights=[2,2,3,3,1,2],
    voting='hard'
)
clf_vote.fit(X,y)

scores = cross_val_score(clf_vote, X, y, cv=5, scoring='accuracy')
print("Voting: Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))

'''
for clf, label in zip(
    [clf_tree, clf_knn, clf_svm, clf_ext, clf_gb, clf_xgb, clf_pctr, clf_log, clf_rf, clf_bag, clf_vote],
    ['tree', 'knn', 'svm', 'extra', 'gb', 'xgb', 'percep', 'logistic', 'RF', 'Bag', 'Ensemble']):
    scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
'''
# Preparing our prediction for submission
clf = clf_vote
df2 = test.loc[:,cols].fillna(method='pad')
surv_pred = clf.predict(df2)
submit = pd.DataFrame({'PassengerId' : test.loc[:, 'PassengerId'],
                       'Survived' : surv_pred.T})
submit.to_csv("submit.csv", index=False)
print(submit.head())
print(submit.shape)