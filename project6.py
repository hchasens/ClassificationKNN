import numpy as np
import sys
import matplotlib.pyplot as plt
from numpy.linalg.linalg import norm
import pandas as pd
from pandas.core.indexes.numeric import Float64Index
import seaborn as sns
from util import *
import math

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

graphing = False         # if set to true it will start plots, this is set to false by default


def main():
    iris = sns.load_dataset("iris")
    
    
    penguins = sns.load_dataset("penguins")
    penguins.pop("island")
    penguins.pop("sex")
    penguins.dropna(axis=1) #drop rows with nan values
    
    print("\npenguins")

    for i in range(1, 10):
        accuracy = 0 
        for x in range(20): #we average 20 tests for each k
            (penguinsTestGuess, penguinsAccuracy) = knn(penguins, k = i)
            accuracy += penguinsAccuracy.iloc[0] #add the true accuracy value
        accuracy = accuracy/20 # averages the accracy of each k over 20 tries
        print("\nk = ", i, "\n accuracy = ", accuracy)
 
    print("\niris")

    for i in range(1, 10):
        accuracy = 0 
        for x in range(20): #we average 20 tests for each k
            (irisTestGuess, irisAccuracy) = knn(iris, k = i)
            accuracy += irisAccuracy.iloc[0] #add the true accuracy value
        accuracy = accuracy/20 # averages the accracy of each k over 20 tries
        print("\nk = ", i, "\n accuracy = ", accuracy)
 


def knn(data, trainRatio = .80, d = 2, k = 5):
    #trainRatio = .80 # the ratio at which the data will be split into both training and test sets. 
    """
    1. Visualize the raw data (e.g. in a pair plot), and form a hypothesis about which classifier will perform better: KNN or Naive Bayes.
        I think that KNN will do better. 
    """
    #sns.pairplot(iris, hue="species")   #original iris pairplot (step 1)
    """
    2. Preprocess as you see fit. For example: normalization, PCA, training/test sets, …
        CAUTION: The mean, std, and PCs should be calculated for the training set only. The test set should be normalized using the training set’s mean & std, and rotated onto the PCs of the training set. Otherwise the test space != training space, and high jinx ensue.

    In this case I'll be training with %80 of my total data
    I'll normalize by z-score
    Then find the PCA for the training set
    """
    suffled = data.sample(frac = 1)  #suffled now contains the shuffled iris dataset
    n = suffled.shape[0]

    train = suffled[0:int(trainRatio*n)]         # sets the training set to $TrainingRatio$ of the suffled data 
    test = suffled[int(trainRatio*n):-1]

    speciesTrain = train.pop("species") #.to_numpy() #.reshape(train.shape[0], 1)
    speciesTest = test.pop("species") #.to_numpy() #.reshape(test.shape[0], 1)

    train = train #.to_numpy()
    test = test #.to_numpy()

    (normTrain, meanTrain, stdTrain) = Utils.z_score(train)
   



    # PCA starts hear
    #d = 2   #dimentions to keep during PCA`
    (pc, eig) = Utils.getPC(normTrain)

    y = normTrain @ pc
    y_proj = y.iloc[:, 0:d]   

    pc = pd.DataFrame(pc)
    
    normalizedPCAdata = (y_proj @ pc.iloc[:,0:d].T)
    normalizedPCAdata.columns = train.columns 
    data_rec = normalizedPCAdata * stdTrain + meanTrain # I do this seperatly so I can add metadata to the normalized PCA data, otherwise it has an issues multipling with the std and mean

    preprosTrain = pd.concat([data_rec, speciesTrain], axis=1) # this is our compleate dataset that has gone through PCA and our preprocessing steps
    #print(preprosTrain)
    if graphing:
        sns.pairplot(preprosTrain, hue="species")           # pair plot of our preprocess data


    # START OF KNN ---- START OF KNN ---- START OF KNN
            # and issue i see is passing the test data through the PCA projection that we've made. I'm not sure how to do that

    #start of preprocessing test data
    normTest = Utils.normalize(test, meanTrain, stdTrain)
    #pca on test data

    y = normTest @ pc.to_numpy()
    y_proj = y.iloc[:, 0:d]

    pc = pd.DataFrame(pc)
 
    normalizedPCAdata = (y_proj @ pc.iloc[:,0:d].T)
    normalizedPCAdata.columns = train.columns 
    preTest = normalizedPCAdata * stdTrain + meanTrain # I do this seperatly so I can add metadata to the normalized PCA data, otherwise it has an issues multipling with the std and mean


    preprosTestCompleated = pd.concat([preTest, speciesTest], axis=1) # this is our PCA space test set with answers!! 

    if graphing:
        sns.pairplot(preprosTestCompleated, hue="species")
    #somethings going wrong with my PCA.

    # for now I'm just gonna create a var named preTest which will be our projected test set (once i figure out how to do that)
    
    # need to find a way to put the test set through PCA




    #k = 1   #how many neighbors to visit

    #start with a point x
    #find the distance from point x to all the other points in train
    #select the first k points, vote, then clasify

    """
    In my first atempt to find the distances I wanted to use numpy.linalg.norm() but this didn't work. hear is my work of my tring to get it to work
        I was gonna try to vectorize this but I wanna get it working first so I'm just gonna use a forloop then try to vecotrise it 

        Here are some notes on my init atempt
                    # to vectorize via numpy/pandas this lets find the distance of everypoint to everyother point
                    #distance = np.linalg.norm(preTest - preTest, axis=1)
        
        #dist = np.empty(1)
        # hear i'm wanna find the distance from preTest[0, :] to every other row in preTest so I want an output the length of preTest for every elemnet in preTest
        #print(np.linalg.norm(preTest.to_numpy()[0, :] - preTest.to_numpy()))
        #print(preTest.to_numpy()[0, :].shape)
        #print(preTest.to_numpy().shape) 

        #print(np.linalg.norm(preTest.to_numpy() - data_rec.to_numpy(), axis = (1,1)))         # this is giving me an error but should, in one line, be able to get distance of every point to every other point
        
        # in lue of this working I'll have to use a for loop and not use a tuple for the axis
        # I need to find the distance of every point x in test to every point i in train
        for x in range(preTest.shape[0]):
            print(preTest.iloc[x,:].shape)
            print(data_rec.to_numpy().shape)
            dist = np.linalg.norm(preTest.iloc[x,:] - data_rec.to_numpy())
            # we now have the distance from point x to every other point. Now we need to sort
            #dist
    """
    # after spending some time trying to get higher dimentional euclidean distance vectorized
    def euclidDist(x, y):
        dist = 0.0
        for i in range(len(x)-1):
            dist += (x[i] - y[i])**2
        return math.sqrt(dist)
    
    dist = np.empty((preTest.shape[0], data_rec.shape[0]))
    for x in range(preTest.shape[0]):
        for i in range(data_rec.shape[0]):
            dist[x, i] = (euclidDist(preTest.iloc[x,:], data_rec.iloc[i,:]))
    #dist is now a 29, 120 shape ndarray that contains the distance from x to every other point in training
    # now we sort to find knn

    knn = np.argsort(dist, axis = 1) # returns the order, from gratest to smallest, in which one can find x's closest neighbor
    #knn = np.flip(knn, axis = 1) #flips the array so that the first colums hold the number of the closest points # as it turns out it doens't need to be flipped. This cost me hours of debugging to relize that it was ordering things properly 
    
    #knn = knnFlipped
    # print(knnFlipped)
    # print(knn)
    # the first colum should be it's self or its 'identity', so the first neighbor should be the index in the second column
    assumedSpecies = pd.Series(dtype=object).reindex_like(speciesTest)
    assumedSpecies.name = "assumedSpecies"
    #print(assumedSpecies)
   
    for x in range(knn.shape[0]):
        p = preTest.iloc[x, :]          # the curent unknown Iris test row
        nn = knn[x, 1:k+1] # nearest neighbors (we add one to avoid the identity column)
        #print("\n at this point \n", p, " its nearest neighbors are at ", nn) 
        #for now lets just go with the nearest neighbor the implement a voting system
        #print("x is ", x, ". NN is ", nn[0])
        #print("speciesTrain at nn is ", speciesTrain.iloc[nn[0]]) #this works but the line bellow it will delever nan s sometimes
        
        poll = speciesTrain.iloc[nn[0:k]]
        #print(poll)
        vote = poll.value_counts()
        #print(vote)
    

        assumedSpecies.iloc[x] = speciesTrain.iloc[nn[0]] # k = 1
        assumedSpecies.iloc[x] = vote.index[0] # k = k
        
        
        
        
        # I figured it out, it was becuase i didn't use .iloc in front of the slice. I figured i didn't need to because it was a Series not a dataframe, i guess i was wrong

        # yay!! we now have an assumed list of species from our test set. Time to test now acruate it is with the actual set!


    accuracyBool = (assumedSpecies == speciesTest)
    accuracyRaw = accuracyBool.value_counts()
    accuracy = accuracyBool.value_counts(normalize=True) #sets the value equal to the ratio of accruacy acheaved
  
    finalTest = pd.concat([test, assumedSpecies], axis=1) # this is our compleate dataset that has gone through PCA and our preprocessing steps
  
    return(finalTest, accuracy)

if __name__=="__main__":
    main()
    plt.show()
