# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:45:36 2017

@author: lbechberger
"""

class Weights:
    """Specifies a set of weights W = <W_delta, {W_d}>.
    
    Weights must always fulfill their normalization constraints."""
    
    def __init__(self, domain_weights, dimension_weights):
        """Initializes the weights and normalizes them if necessary.
        
        'domain_weights' is a mapping from domains to weights and 
        'dimension_weights' contains for each domain a mapping from dimensions to weights."""
        
        self.domain_weights = self._normalize(domain_weights, len(domain_weights.keys()))
        self.dimension_weights = {}
        for (domain, weights) in dimension_weights.items():
            self.dimension_weights[domain] = self._normalize(weights, 1.0)
        
    def _normalize(self, weights, total):
        """Normalizes a given set of weights such that they sum up to the desired total."""
        
        result = {}
        old_sum = sum(weights.values())
        
        for (k,v) in weights.items():
            result[k] = (1.0*v*total)/(old_sum)
        
        return result

    def _check(self, domain_weights = None, dimension_weights = None):
        """Checks if all normalization constraints are fulfilled."""

        domain_weights = domain_weights if domain_weights != None else self.domain_weights        
        dimension_weights = dimension_weights if dimension_weights != None else self.dimension_weights        
        
        if sum(domain_weights.values()) != len(domain_weights.keys()):
            return False
        
        for weights in dimension_weights.values():
            if sum(weights.values()) != 1.0:
                return False
        
        return True
    
    def __str__(self):
        return "w<{1},<{2}>>".format(str(self.domain_weights),str(self.dimension_weights))