# -*- coding: utf-8 -*-
"""
Created on Wed May  8 14:31:31 2019

@author: zhshah
"""

from __future__ import division
import numpy as np
import cma
import heapq 


   
def ObjectiveFunction(proposedImage, originalImage, trueLabel, model):
    predictedLabel=model.predict([proposedImage])
    if predictedLabel==trueLabel:
        probabilities=model.predict_proba([proposedImage])
        probabilities=list(probabilities[0])
        probab=heapq.nlargest(2, probabilities)
        probabilitydifference=probab[0]-probab[1] 
        return probabilitydifference 
    else:
        euclideandistance=-1.0/np.linalg.norm(originalImage - proposedImage)
        return euclideandistance
    



def Optimizer(sigma, objectivefunc, proposedImage, originalImage, trueLabel, model):
    if len(proposedImage) is not 0:
        result=cma.fmin(objectivefunc, proposedImage, sigma, options={'tolx':0.01}, args=(originalImage, trueLabel, model))
        return result
    else:
        print ('The proposed image is missing')
