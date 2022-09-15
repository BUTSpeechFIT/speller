#!/usr/bin/python

import numpy as np

def CMVN(x,norm_means=True,norm_vars=True):
        mean = x.mean(axis=0)
        square_sums = (x ** 2).sum(axis=0)

        var = square_sums / x.shape[0] - mean ** 2
        std = np.maximum(np.sqrt(np.abs(var)),1.0e-20)
        
        #-------------------------------
        if norm_means:
                x = np.subtract(x, mean)
        if norm_vars:
                x = np.divide(x, std)
        #-------------------------------
        return x

