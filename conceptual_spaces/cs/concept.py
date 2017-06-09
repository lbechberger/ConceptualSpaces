# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: lbechberger
"""

from math import exp

import core as cor
import weights as wghts
import cs

class Concept:
    """A concept, implementation of the Fuzzy Simple Star-Shaped Set (FSSSS)."""
    
    def __init__(self, core, mu, c, weights):
        """Initializes the concept."""

        if (not isinstance(core, cor.Core)) or (not core._check()):
            raise Exception("Invalid core")

        if mu > 1.0 or mu <= 0.0:
            raise Exception("Invalid mu")
        
        if c <= 0.0:
            raise Exception("Invalid c")
        
        if (not isinstance(weights, wghts.Weights)) or (not weights._check()):
            raise Exception("Invalid weights")
        
        self._core = core
        self._mu = mu
        self._c = c
        self._weights = weights
            
    def __str__(self):
        return "<{0},{1},{2},{3}>".format(self._core, self._mu, self._c, self._weights)
    
    def __eq__(self, other):
        if not isinstance(other, Concept):
            return False
        if self._core != other._core or self._mu != other._mu or self._c != other._c or self._weights != other._weights:
            return False
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def membership(self, point):
        """Computes the membership of the point in this concept."""
        
        min_distance = reduce(min, map(lambda x: cs.ConceptualSpace.cs.distance(x, point, self._weights), self._core.find_closest_point_candidates(point)))
        
        return self._mu * exp(-self._c * min_distance)
    
    def intersect(self, other):
        """Computes the intersection of two concepts."""
        pass #TODO implement

    def unify(self, other):
        """Computes the union of two concepts."""

        if not isinstance(other, Concept):
            raise Exception("Not a valid concept")
        
        core = self._core.unify(other._core) 
        mu = max(self._mu, other._mu)
        c = min(self._c, other._c)
        weights = self._weights.merge(other._weights, 0.5, 0.5)
        
        return Concept(core, mu, c, weights)
        
        
    def project(self, domains):
        """Computes the projection of this concept onto a subset of domains."""
        
        # no explicit check for domains - Core will take care of this
        new_core = self._core.project(domains)
        new_weights = self._weights.project(domains)
        
        return Concept(new_core, self._mu, self._c, new_weights)

    def cut(self, dimension, value):
        """Computes the result of cutting this concept into two parts (at the given value on the given dimension).
        
        Returns the lower part and the upper part as a tuple (lower, upper)."""
        
        lower_core, upper_core = self._core.cut(dimension, value)
        lower_concept = None if lower_core == None else Concept(lower_core, self._mu, self._c, self._weights)
        upper_concept = None if upper_core == None else Concept(upper_core, self._mu, self._c, self._weights)
        
        return lower_concept, upper_concept

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