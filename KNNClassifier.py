# -*- coding: utf-8 -*-
"""
Created on Thu May  9 17:23:42 2019

@author: zhshah
"""

from sklearn.neighbors import KNeighborsClassifier
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


def KnnModel(trainImages, trainLabel, testImages):
    print ("Training Knn Classifier")
    model =KNeighborsClassifier(algorithm='ball_tree', leaf_size=30, metric='euclidean',
           metric_params=None, n_jobs=None, n_neighbors=10, p=2,
           weights='distance')
    model.fit(trainImages,trainLabel)
    predlabel= model.predict(testImages)
    return predlabel, model



predictedLabel, KnnModel = KnnModel(trainImages, trainLabels, testImages)
accuracy= accuracy_score (testLabels, predictedLabel)
print('The accuracy of K-Nearest Neigbhors Classifier is', accuracy)

AdversarailInputsKnn=[]

################# Calling function from Generic Method Module & Generating Adversarial Inputs in Knn#############################
for i in range(0, len(testLabels)):
    try:
        res=Optimizer(sigma,ObjectiveFunction, testImages[i], testImages[i], testLabels[i], KnnModel)
        AdversarailInputsKnn.append(res[0])
    except Exception as e:
        print ('Error is :',e)
        pass
                        
##############################################################################################################################
    
print (len(AdversarailInputsKnn), 'number of inputs are converted into adversarial inputs from the Total inputs ', len(testLabels), 'by using K-Nearest Neigbhors')     