import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='sklearn')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

#Understanding data using descriptive statistics and visualization.
url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names=['sepal_length','sepal_width','petal_length','petal_width','class']
dataset=pd.read_csv(url,names=names)
for column in dataset.columns:
    if dataset[column].dtype == type(object):
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
array=dataset.values
scatter_matrix(dataset)
plt.show()
plt.plot(dataset.iloc[:,[0,1,2,3]],label='names[]')
X=array[:,0:4]
Y=array[:,4]

#Pre-Processing the data to best expose the structure of the problem.
scaler=Normalizer().fit(X)
rescaled=scaler.transform(X)
np.set_printoptions(precision=3)

#Checking a number of algorithms using own test harness.
scoring='accuracy'
models=[]
results=[]
names=[]
models.append(('SVC',SVC()))
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
print('The result after spot-checking is :\n')
for name, model in models:
    kfold = KFold(n_splits=10, random_state=7)
    cv_results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

#Improving results using algorithm parameter tuning.
alphas=np.array([1,0.1,.01,.001,.0001,0])
param_grid=dict(alpha=alphas)
model2=Ridge()
grid=GridSearchCV(estimator=model2,param_grid=param_grid,cv=5,refit=True,error_score=0,n_jobs=-1)
grid.fit(X,Y)
print('\nThe result after parameter tuning:')
print(grid.best_score_)
print(grid.best_estimator_.alpha)

#Improving results using ensemble methods.
print('\nThe results after using ensemble methods :')
num_trees=100
max_features=3
model3=RandomForestClassifier(n_estimators=num_trees,max_features=max_features)
result2=cross_val_score(model3,X,Y,cv=kfold)
print(result2.mean())

# Finalize the model ready for future use.
test_size = 0.33
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
model = LogisticRegression()
model.fit(X_train, Y_train)
filename = 'finalized1_model.sav'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test, Y_test)
print('\nThe result after Model is ready for future use is :\n')
print(result)