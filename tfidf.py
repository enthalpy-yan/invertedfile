"""
Module for TF-IDF
"""
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
