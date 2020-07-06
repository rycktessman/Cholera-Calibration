#####################################################################################
# Author: Theresa Ryckman (tessryckman@gmail.com)                                   #
# Purpose: used to induce parameter correlation from LHS samples in calibration     #
# Last Updated: June 8, 2020                                                        #
#####################################################################################

import numpy as np

# Function inputs:
# 1. X: matrix of uncorrelated parameter draws (n samples by 4 parameters, in this case) to induce correlation
# 2. normals: matrix of multinormal samples (same dimensions) used to induce correlation on X

def sorted_rank(X, normals):
    col=np.shape(normals)[1]
    n=np.shape(normals)[0]
    Xsorted=np.zeros((n, col))
    Nrank=np.zeros((n))
    Nrank=Nrank.astype(int)
    Xstar=np.zeros((n, col))
    for j in range(0, col):
        Xsorted[:,j]=np.sort(X[:,j])
        Nrank[np.argsort(normals[:,j])]=np.arange(0,n)
        Xstar[:,j]=Xsorted[Nrank[::1],j]
    return Xstar