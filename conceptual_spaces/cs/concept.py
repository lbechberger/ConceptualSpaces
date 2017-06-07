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
    
    def intersect(self, other):
        """Computes the intersection of two concepts."""
        pass #TODO implement

    def unify(self, other):
        """Computes the union of two concepts."""
        pass #TODO implement
        
    def project(self, domains):
        """Computes the projection of this concept onto a subset of domains."""
        pass #TODO implement

    def cut(self, dimension, value):
        """Computes the result of cutting this concept into two parts (at the given value on the given dimension)."""
        pass #TODO implement

    def hypervolume(self):
        """Computes the hypervolume of this concept."""
        pass #TODO implement

    def subset_of(self, other):
        """Computes the degree of subsethood between this concept and a given other concept."""
        pass #TODO implement

    def implies(self, other):
        """Computes the degree of implication between this concept and a given other concept."""
        pass #TODO implement
    
    def similarity(self, other):
        """Computes the similarity of this concept to the given other concept."""
        pass #TODO implement

    def between(self, first, second):
        """Computes the degree to which this concept is between the other two given concepts."""
        pass #TODO implement