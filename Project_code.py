########################## Importing libraries ###############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

########################## Reading the file #################################
data_bank1 = pd.read_csv('bank1.csv',sep =';')
# data_bank = pd.read_csv("bank.csv",sep=';')
data_bank1.head()
iv = data_bank1.iloc[:,0:20]
dv = data_bank1.iloc[:,20]

########################### checking for missing values ######################
data_bank1.isnull().sum()

########################### Visualization ####################################
sns.countplot(y='y', data=data_bank1)
#Visualizing the predicting varibale we can see that the data is imbalenced.

###################### correlation ###########################################
iv_corr = iv.corr()

#sorting the correlation
ic = iv_corr.abs().unstack()
so = ic.sort_values(kind="quicksort")
# plot the heatmap
sns.heatmap(iv_corr,cmap="YlGnBu")

############## Removing the correlated variables ############################
new_iv = iv.drop(['euribor3m','emp.var.rate'],axis=1)

########################### Checking the datatypes ###########################
dtype_df = new_iv.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()
       
#################### Changing catogotical variables #########################
iv_one_hot = pd.get_dummies(new_iv)
from sklearn.preprocessing import LabelEncoder
labelencoder_dv = LabelEncoder()
dv_one_hot = labelencoder_dv.fit_transform(dv)

#Checking the datatypes again ensuring there is no catogorical data
dtype_df = iv_one_hot.dtypes.reset_index()
dtype_df.columns = ["Count", "Column Type"]
dtype_df.groupby("Column Type").aggregate('count').reset_index()    

####### Feature importance using Random Forest ##############################

forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)
forest.fit(iv_one_hot, dv_one_hot)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

for f in range(iv_one_hot.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(iv_one_hot.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(iv_one_hot.shape[1]), indices)
plt.xlim([-1, iv_one_hot.shape[1]])
plt.show()

###################### Feature Scalling #####################################
sc_iv = StandardScaler()
iv_scale = sc_iv.fit_transform(iv_one_hot)

################# Split the dataset into train and test dataset ##############
iv_train, iv_test, dv_train ,dv_test = train_test_split(iv_scale,
                            dv_one_hot, test_size =0.25,random_state = 0)


################# Training the model ########################################
# Building different model to choose the best model with cross fold validation

models = []
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5,
                                           metric = 'minkowski', p = 2)))
models.append(('Decison-Tree', DecisionTreeClassifier(criterion = 'entropy')))
models.append(('SVM', SVC()))
models.append(('RandForest',RandomForestClassifier(criterion = 'entropy',
                                                   n_estimators = 10 )))


results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits=10, random_state=0)
    cv_results = model_selection.cross_val_score(model, iv_scale, dv_one_hot,
                                                 cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "{}: {}".format(name, cv_results.mean())
    print(msg)

sns.set(rc={'figure.figsize':(10,8)})
sns.boxplot(names,results)

# From the result we can see that Logistic regression and LDA provides a better model 
# with more accuracy and with less run time

# Using Logestic regression to predict the test data
logit_classifier = LogisticRegression() 
logit_classifier.fit(iv_train, dv_train)
predict_logit = logit_classifier.predict(iv_test)
print("\n****************** LOGESTIC REGRESSION ********************* \n")
print("Accuracy : ", accuracy_score(dv_test, predict_logit))
print("Confusion Matrix : \n",confusion_matrix(dv_test, predict_logit))
print("Classification Report: \n",classification_report(dv_test, predict_logit))

# Using LDA to predict the test data
lda_classifier = LinearDiscriminantAnalysis() 
lda_classifier.fit(iv_train, dv_train)
predict_lda = lda_classifier.predict(iv_test)
print("\n****************** LDA MODEL ********************* \n")
print("Accuracy : ", accuracy_score(dv_test, predict_lda))
print("Confusion Matrix : \n",confusion_matrix(dv_test, predict_lda))
print("Classification Report: \n",classification_report(dv_test, predict_lda))

