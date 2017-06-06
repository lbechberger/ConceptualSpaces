# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: lbechberger
"""

from core import Core

class Concept:
    """A concept, implementation of the Fuzzy Simple Star-Shaped Set (FSSSS)."""
    
    def __init__(self, core, mu, c, weights):
        """Initializes the concept."""

        if (not isinstance(core, Core)) or (not core._check()):
            raise Exception("Invalid core")

        if mu > 1.0 or mu <= 0.0:
            raise Exception("Invalid mu")
        
        if c <= 0.0:
            raise Exception("Invalid c")
        
        #TODO: check weights!
        
        self._core = core
        self._mu = mu
        self._c = c
        self._weights = weights
    
    def __str__(self):
        return "<{0},{1},{2},{3}>".format(self._core, self._mu, self._c, self._weights)
    
    