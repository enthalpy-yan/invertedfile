"""
Module for TF-IDF
"""

import numpy as np
import math
import operator as opt
from itertools import imap, ifilter

def ham_dist(str1, str2):
    """
    Calculate hamming distance between two binary digits string.
    """
    return sum(imap(opt.ne, str1, str2))

def hmatchs(u_iter, v, dist):
    """
    Get a iterator of the list of hamming distances smaller than
    the given hamming distance.

    Parameters
    ----------
    u_iter: list of binary string.
    v: input binary string.
    dist: the largest length of hamming distance.

    Returns
    -------
    A hamming distance iterator.
    [(dist, binary string from original iterator), ....]
    """
    def _get_dist(target):
        return ham_dist(target, v), target
    return ifilter(lambda t: t[0] < dist, imap(_get_dist, u_iter))

def getidf(arr):
    k,l = arr.shape
    idf = []
    j = 0
    for i in arr:
        idf.append(math.log10( l/sum([1 for p in i if p>0])))
    return idf

def tf_idf(txt,k):
    f = open(txt,'r')
    l = sum(1 for line in f)
    nparr = np.zeros((l,k))
    f = open(txt,'r')
    # calculate tf
    j = 0
    for line in f:
        value = [int(i.split(":")[1]) for i in line.split(" ")[1:-1]]
        value = [ float(i)/sum(value) for i in value]
        nparr[j] = np.asarray(value)
        j += 1

    idf = getidf(nparr.T )

    #calculate tf-idf
    for i in range(l):
        nparr[i] = nparr[i]*idf

    return idf,nparr.T
