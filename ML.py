#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 12:44:06 2017

@author: sahana
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:54:06 2017

@author: sahana
"""
#import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

#iris = datasets.load_iris()
#X = iris.data[:, 0:2]  # we only take the first two features for visualization
#y = iris.target
df = pd.DataFrame.from_csv('ITPFULLDATA.csv',index_col='Study_ID')
print(df.head())
#df.columns = ["Pain",	"Soreness", "Aching",	"Tenderness",	"Arm or hand swelling",	"Breast swelling",	"Chest wall swelling","Firmness in the affected limb",	"Tightness in the affected limb",	"Heaviness in the affected limb",	"Toughness or thickness of skin in the affected limb","Stiffness in the affected limb",	"Hotness/increased temperature in the affected limb",	"Redness in the affected limb","	Blistering in the affected limb",	"Numbness in the affected limb","	Burning in the affected limb",	"Stabbing in the affected limb","	Tingling (pins and needles) in the affected limb","	Fatigue in the affected limb","	Weakness in the affected limb",	"Pocket of fluid developed",	"Result"] 


df.columns = ["Symptom_Pain","	Symptom_Soreness	","Symptom_Aching	","Symptom_Tenderness	","Symptom_Arm_Hand_Swelling	","Symptom_Breast_Swelling	","Symptom_Chest_Wall_Swelling	","Symptom_Firmness	","Symptom_Tightness","	Symptom_Heaviness	","Symptom_Toughness	","Symptom_Stiffness	","Symptom_Hotness/Increased_Temperature	","Symptom_Redness	","Symptom_Blistering	","Symptom_Numbness	","Symptom_Burning","	Symptom_Stabbing","	Symptom_Tingling	","Symptom_Fatigue	","Symptom_Weakness	","Symptom_Pocket_Fluid	","Limited_Movement_Shoulder	","Limited_Movement_Elbow	","Limited_Movement_Wrist	","Limited_Movement_Fingers	","Limited_Movement_Arm	","Sentinel_Lymph_0des_Biopsy	","Axillary_Lymph_0des_Dissection(ALND)",	"Radiation(Yes/0)",	"Chemotherapy(Y/N)", "Target_Zone"]
#df.plot()
#plt.show()
X = df.drop('Target_Zone', 1)
print(X.head())
y = df.Target_Zone
print(y)
n_features = X.shape[1]

C = 1.0
kernel = 1.0 * RBF([1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0])  # for GPC

# Create different classifiers. The logistic regression cannot do
# multiclass out of the box.
classifiers = {'L1 logistic': LogisticRegression(C=C, penalty='l1'),
               'L2 logistic (OvR)': LogisticRegression(C=C, penalty='l2'),
               'Linear SVC': SVC(kernel='linear', C=C, probability=True,
                                 random_state=0),
               'L2 logistic (Multinomial)': LogisticRegression(
                C=C, solver='lbfgs', multi_class='multinomial'),
               'GPC': GaussianProcessClassifier(kernel)
               }
n_classifiers = len(classifiers)

#xx = np.linspace(3, 9, 100)
#yy = np.linspace(1, 5, 100).T
#xx, yy = np.meshgrid(xx, yy)
#Xfull = np.c_[xx.ravel(), yy.ravel()]
for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X, y)
    y_true = y
    target_names = ['class 0', 'class 1', 'class 2','class 3','class 4','class 5', 'class 6', 'class 7','class 8']
    y_pred = classifier.predict(X)
    classif_rate = np.mean(y_pred.ravel() == y.ravel()) * 100
    print("classif_rate for %s : %f " % (name, classif_rate))
    print(confusion_matrix(y_true, y_pred) )
    print(classification_report(y_true, y_pred, target_names=target_names))
   

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=2)
    metrics.auc(fpr, tpr)
    print(metrics.auc(fpr, tpr))
#    sample = [1,3,0,0,1,2,3,4,1,2,4,0,0,0,0,1,1,0,0,1,1,2]
#    result = classifier.predict(sample)
##Check the result
#    print(result[0])
    # View probabilities=
#    probas = classifier.predict_proba(Xfull)
#    n_classes = np.unique(y_pred).size
#  
sample =[2,2,2,2,2,2,1,2,3,2,0,3,0,0,0,1,0,0,2,2,4,0,2,0,0,0,2,0,1,1,1]
#sample = [1,3,0,0,1,2,3,4,1,2,4,0,0,0,0,1,1,0,0,1,1,2]
result = classifier.predict(sample)
#Check the result
print(result)
###