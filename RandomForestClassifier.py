# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:39:02 2019

@author: zhshah
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import GradientBasedMethodSVM
from GenericMethod import ObjectiveFunction, Optimizer


###################Parameters Loading data from other module ###########################
trainImages=GradientBasedMethodSVM.train_x
trainLabels=GradientBasedMethodSVM.train_y
testImages=GradientBasedMethodSVM.test_x
testLabels=GradientBasedMethodSVM.test_y
sigma=0.1


##########################################Random Forest Model ############################

def RandomForestModel(trainImages, trainLabel, testImages):
    print ("Training Random Forest Classifier")
    model =RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=None, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=1, min_samples_split=2,
            min_weight_fraction_leaf=0.0, n_estimators=50, n_jobs=None,
            oob_score=False, random_state=None, verbose=0, warm_start=True)
    model.fit(trainImages,trainLabel)
    predlabel= model.predict(testImages)
    return predlabel, model



predictedLabel, trainedModel = RandomForestModel(trainImages, trainLabels, testImages)
accuracy= accuracy_score (testLabels, predictedLabel)
print('The accuracy of random forest is', accuracy)

AdversarailInputsRandomForest=[]

################# Calling function from Generic Method Module & Generating Adversarial Inputs#############################
for i in range(0, len(testLabels)):
    try:
        predictedLabel=trainedModel.predict([testImages[i]])
        if predictedLabel == testLabels[i]:
            res=Optimizer(sigma,ObjectiveFunction, testImages[i], testImages[i], testLabels[i], trainedModel)
            AdversarailInputsRandomForest.append(res[0])
        else:
            continue
    except Exception as e:
        print ('Error is :',e)
        pass
                        
##############################################################################################################################
    
print (len(AdversarailInputsRandomForest), 'number of inputs are converted into adversarial inputs from the Total inputs ', len(testLabels))     