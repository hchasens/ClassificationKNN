import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.linalg.linalg import norm
import pandas as pd
import seaborn as sns
from util import *

__author__ = "Hunter Chasens"
__license__ = "GPLv3"
__version__ = "0.1"
__email__ = "hunter.chasens18@ncf.edu"

"""
To Do
1. Visualize the raw data (e.g. in a pair plot), and form a hypothesis about which classifier will perform better: KNN or Naive Bayes.

2. Preprocess as you see fit. For example: normalization, PCA, training/test sets, …
    CAUTION: The mean, std, and PCs should be calculated for the training set only. The test set should be normalized using the training set’s mean & std, and rotated onto the PCs of the training set. Otherwise the test space != training space, and high jinx ensue.

3. Apply your own implementation of KNN to Iris + one other dataset of your choice.
    Implement & apply Naive Bayes, as well, if you have time, for those last 3% and a well-deserved sense of accomplishment.

4. What's the best number of neighbors, K, for your dataset? Defend your answer quantitatively.

5. Evaluate the performance of your classifier(s). Defend your answer quantitatively, using confusion matrices and whichever of the following matter most in your application:
    true positive (TP) rate,
    false positive (FP) rate, and/or
    precision.

6. What did you learn about the classes in your chosen dataset by developing and evaluating these classifiers?
"""

sns.set_theme(style="ticks")

def knn():
    pass 



def main():
    trainRatio = .80
    d = 2   #dimentions to keep during PCA`
    iris = sns.load_dataset("iris")
    """
    1. Visualize the raw data (e.g. in a pair plot), and form a hypothesis about which classifier will perform better: KNN or Naive Bayes.
        I think that KNN will do better. 
    """
    sns.pairplot(iris, hue="species")   #original iris pairplot (step 1)
    """
    2. Preprocess as you see fit. For example: normalization, PCA, training/test sets, …
        CAUTION: The mean, std, and PCs should be calculated for the training set only. The test set should be normalized using the training set’s mean & std, and rotated onto the PCs of the training set. Otherwise the test space != training space, and high jinx ensue.

    In this case I'll be training with %80 of my total data
    I'll normalize by z-score
    Then find the PCA for the training set
    """
    suffled = iris.sample(frac = 1)  #suffled now contains the shuffled iris dataset
    n = suffled.shape[0]

    train = suffled[0:int(trainRatio*n)]         # sets the training set to $TrainingRatio$ of the suffled data 
    test = suffled[int(trainRatio*n):-1]

    speciesTrain = train.pop("species") #.to_numpy() #.reshape(train.shape[0], 1)
    speciesTest = test.pop("species") #.to_numpy() #.reshape(test.shape[0], 1)

    train = train #.to_numpy()
    test = test #.to_numpy()

    (normTrain, meanTrain, stdTrain) = Utils.z_score(train)
   
    # PCA starts hear
    (pc, eig) = Utils.getPC(normTrain) 
    y = normTrain @ pc
    y_proj = y.iloc[:, 0:d]   

    pc = pd.DataFrame(pc)
    
    normalizedPCAdata = (y_proj @ pc.iloc[:,0:d].T)
    normalizedPCAdata.columns = train.columns 
    data_rec = normalizedPCAdata * stdTrain + meanTrain # I do this seperatly so I can add metadata to the normalized PCA data, otherwise it has an issues multipling with the std and mean
   
    preprosTrain = pd.concat([data_rec, speciesTrain], axis=1) # this is our compleate dataset that has gone through PCA and our preprocessing steps
    print(preprosTrain)
    sns.pairplot(preprosTrain, hue="species")


    # START OF KNN ---- START OF KNN ---- START OF KNN

    


if __name__=="__main__":
    main()
    plt.show()
