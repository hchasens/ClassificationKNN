""" Contains utility classes.
"""

import numpy as np
import pandas as pd

__author__ = "Hunter Chasens"
__license__ = "GPLv3"
__version__ = "0.1"
__email__ = "hunter.chasens18@ncf.edu" 


class Utils:
    """[Uninstantiated class that contains utility functions.]
    """

    @staticmethod
    def getPC(normdata):
        """[Returns the order principle compoenets of a dataset. Principle compoenets are order from by their eigenvalues from greatest to least.]

        Args:
            normdata ([nparray]): [data to be processed]

        Returns:
            [nparray, nparray]: [principle compoenets sorted by corresponding eigenvalues from greatest to least, ordered eigenvalues]
        """
        covmrx = np.cov(normdata,rowvar=False)      #creates a coverance matrix out of the normalized data
        (eigvals, pc) = np.linalg.eig(covmrx)       #find the eigenvalues and all unsorted principle compoenets
        order = np.argsort(eigvals)[::-1]           #finds the order from greatest to least of the eigenvalues
        eigvals = eigvals[order]                    #rearranges the eigenvalues from greatest to least
        pc = pc[:, order]                           #rearranges all principle compoenets such that their corresponding eigenvalues are order from greatest to least
        return pc, eigvals

    @staticmethod
    def z_score(data, removeOutliers=True):
        """[normalizes an nparray by z-score (e.g. normalizes all features by standard deviation such that the standard deviation of any feature is 1)]

        Args:
            data ([nparray]): [the nparray to be normalized]
            removeOutliers ([Boolean]): [removes outliers greater then three standard deviations]


        Returns:
            [nparray]: [a Z-Scored normalized nparray]
            [int]: [original mean of data]
            [int]: [original standard deviation of data]

        """
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)   
        mean = np.mean(data)
        std = np.std(data)      
        zscore = ( data - mean ) / std  # I keep getting a bug here, I think its because some std values are 0 so its trying to divide by zero
        if removeOutliers == True:
            zscore = zscore[np.any(zscore < 3*std, axis=1)]
        return zscore, mean, std

    @staticmethod
    def normalize(data, mean, std, removeOutliers=True):
        """[normalizes an nparray by z-score (e.g. normalizes all features by standard deviation such that the standard deviation of any feature is 1)]

        Args:
            data ([nparray]): [the nparray to be normalized]
            removeOutliers ([Boolean]): [removes outliers greater then three standard deviations]


        Returns:
            [nparray]: [a Z-Scored normalized nparray]
            [int]: [original mean of data]
            [int]: [original standard deviation of data]

        """
        #Here we are removing the row in which std is zero, meaning all data in that colum is the same. I'm sure there's a cleaner way of doing this and will research it.
        zscore = ( data - mean ) / std  # I keep getting a bug here, I think its because some std values are 0 so its trying to divide by zero
        if removeOutliers == True:
            zscore = zscore[np.any(zscore < 3*std, axis=1)]
        return zscore

    @staticmethod
    def parse(filename):
        """[Converts diffrent files into nparrays. Does not take headers. Only CSVs, TRA, and NPY files as of Alpha]

        Args:
            filename ([String]): [filepath to data, must include filename and extension]

        Returns:
            [nparray]: [the parsed datafile]
        """
        arr = np.empty
        if (".npy" in filename):
            print("reading npy")
            arr = np.load(filename)
        else:
            print("reading csv")
            arr = np.genfromtxt(filename, delimiter=',')
            arr = arr[~np.isnan(arr).any(axis=1)]
        return arr



    @staticmethod
    def pcaPercent(data, percent=80):
        """[PCA by Percent]
        Args:
            data ([nparray]): [input data]
            percent (int, optional): [percent of information retention]. Defaults to 80.
        Returns:
            [nparray]: [nparray containing Principle Components with the <percent> information retained]
        """
        assert(percent <= 100 and percent >= 0)
        (norm, mean, std) = z_score(data, False)      #1. Normalize by Z-Score
        (pc, eig) = getPC(norm)                #2. Calculate Principal Components
        y = norm @ pc                                #3. Rotate onto Principal Components
        scaled_eig = (eig/np.sum(eig))*100               # Here we find the percentage of each Principal Component
        d = 0
        sum = 0
        for x in scaled_eig:
            sum += x
            d+=1
            if sum > percent:
                break
        y_proj = y[:,0:d]                            #4. Project onto the d-dimensional subspace defined by the first d principal component
        data_rec = (y_proj @ pc[:,0:d].T)*std + mean #5. Reconstruct
        return data_rec, eig, scaled_eig, d, y_proj




    @staticmethod
    def pca(data, d=2):
        """[PCA]
        Args:
            data ([nparray]): [input data]
            d (int, optional): [dimensional subspace to keep]. Defaults to 2.
        Returns:
            [nparray]: [nparray containing Principle Components with the highest information retained up to <d> dimensions]
        """
        (norm, mean, std) = z_score(data, False)      #1. Normalize by Z-Score
        (pc, eig) = getPC(norm)                #2. Calculate Principal Components
        y = norm @ pc                                #3. Rotate onto Principal Components
        y_proj = y[:,0:d]                            #4. Project onto the d-dimensional subspace defined by the first d principal component
        data_rec = (y_proj @ pc[:,0:d].T)*std + mean #5. Reconstruct
        return data_rec, eig