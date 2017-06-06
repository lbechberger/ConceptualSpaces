# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:15:30 2017

@author: lbechberger
"""

from math import sqrt

class ConceptualSpace:
    """The overall conceptual space.
    
    Should be used as a singleton. Provides some utility functions."""
    
    def __init__(self, n_dim, domains):
        """Initializes a conceptual space with the given numer of dimensions and the given set of domains.
        
        'n_dim' is an integer >= 1 and 'domains' is a dictionary from domain ids to sets of dimensions."""

        if n_dim < 1:
            raise Exception("Need at least one dimension")
        
        if not self._check_domain_structure(domains, n_dim):
            raise Exception("Invalid domain structure")
            
        self._n_dim = n_dim
        self._domains = domains
        
    def _check_domain_structure(self, domains, n_dim):
        """Checks whether the domain structure is valid."""

        vals = [val for domain in domains.values() for val in domain] # flatten values
       
        # each dimension must appear in exactly one domain
        for i in range(n_dim):
            if vals.count(i) != 1:
                return False
        
        # we need the correct number of dimensions in total
        if len(vals) != n_dim:
            return False
        
        # there are no empty domains allowed
        for (k,v) in domains.items():
            if v == []:
                return False
        
        return True
        
    def distance(self, x, y, weights):
        """Computes the combined metric d_C(x,y,W) between the two points x and y using the weights in 'weights'."""
        
        if len(x) != self._n_dim or len(y) != self._n_dim:
            raise Exception("Points have wrong dimensionality")
        
        distance = 0.0

        for domain in self._domains.keys():
            inner_distance = 0.0
            for dimension in self._domains[domain]:
                inner_distance += weights.dimension_weights[domain][dimension] * (x[dimension] - y[dimension])**2
            distance += weights.domain_weights[domain] * sqrt(inner_distance)
        
        return distance