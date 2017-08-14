#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 11:48:33 2017

@author: jan
"""

import itertools
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

#%%

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          size=(5,4)):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=size)    
    plt.imshow(cm*0, interpolation='nearest', cmap="hot_r")
    plt.title(title)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    x=[]
    y=[]
    c=[]
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i,j] > 0:
            x.append(j)
            y.append(i)
            c.append(cm[i,j])
         
    plt.scatter(x,y,c=c, cmap="gist_heat_r",norm=matplotlib.colors.LogNorm(), 
                vmin=np.min(c), vmax=1, s = 2000./(len(classes)**0.7))
    
    plt.colorbar(aspect=30)
    
    np.fill_diagonal(cm,0)
    
    x_labels= [c+" {:.4f}".format(s) for c,s in zip(classes,np.sum(cm,axis=0))]
    y_labels= [c+" {:.4f}".format(s) for c,s in zip(classes,np.sum(cm,axis=1))]
    
    plt.grid()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, x_labels, rotation=90)
    plt.yticks(tick_marks, y_labels)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig("confusion"+str(len(classes))+".png",bbox_inches='tight')

#%%


def load_predictions_plot_confusion(folder, size=(5,4)):
    """load prediction vectors produced by get_prediction.py and plot a 
    confusion matrix using a colorcoded representation of matrix entries"""

    files = os.listdir(folder)
    classes = []
    
    for f in files:
        if f[:8] == "vectors_":
            classes.append(f[8:])
            
    classes.sort()
    
    y_true = []
    y_pred = []
    
    for num,c in enumerate(classes):
        vectors=np.loadtxt(folder+"/vectors_"+c)
        print(num, len(vectors))
        for vector in vectors:
            y_true.append(num)
            y_pred.append(np.argmax(vector))
            
    cnf_matrix = confusion_matrix(y_true, y_pred)
    
    plot_confusion_matrix(cnf_matrix.copy(), classes=classes, normalize=True,
                          title='Normalized confusion matrix',size=size)


#%%
folder = "predictions_voxforge"
load_predictions_plot_confusion(folder)
#%%

folder = "predictions_176/predictions"
load_predictions_plot_confusion(folder,size=(37,30))


