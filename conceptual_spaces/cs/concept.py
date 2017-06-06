# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: lbechberger
"""

from core import Core
from weights import Weights
from math import exp

class Concept:
    """A concept, implementation of the Fuzzy Simple Star-Shaped Set (FSSSS)."""
    
    def __init__(self, core, mu, c, weights, cs):
        """Initializes the concept."""

        if (not isinstance(core, Core)) or (not core._check()):
            raise Exception("Invalid core")

        if mu > 1.0 or mu <= 0.0:
            raise Exception("Invalid mu")
        
        if c <= 0.0:
            raise Exception("Invalid c")
        
        if (not isinstance(weights, Weights)) or (not weights._check()):
            raise Exception("Invalid weights")
        
        self._core = core
        self._mu = mu
        self._c = c
        self._weights = weights
        self._cs = cs
    
    def __str__(self):
        return "<{0},{1},{2},{3}>".format(self._core, self._mu, self._c, self._weights)
    
    def membership(self, point):
        """Computes the membership of the point in this concept."""
        
        min_distance = reduce(min, map(lambda x: self._cs.distance(x, point, self._weights), self._core.find_closest_point_candidates(point)))
        
        return self._mu * exp(-self._c * min_distance)
    
    