"""
Module for TF-IDF
"""

from itertools import imap
from scipy import spatial

def hmatchs(u_iter, v):
    """
    Calculate hamming distance between the input binary string and string
    iterator.

    Parameters
    ----------
    u_iter: list of binary string.
    v: input binary string.

    Returns
    -------
    A sorted hamming distance list.
    [(dist, binary string from original iterator), ....]
    """
    def _get_dist(target):
        return spatial.distance.hamming(target, v), target
    return sorted(imap(_get_dist, u_iter))
