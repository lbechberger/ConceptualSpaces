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
        return "w<{0},<{1}>>".format(str(self.domain_weights),str(self.dimension_weights))
    
    def __eq__(self, other):
        if not isinstance(other, Weights):
            return False
        if not self.dimension_weights == other.dimension_weights:
            return False
        if not self.domain_weights == other.domain_weights:
            return False
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def merge(self, other, s = 0.5, t = 0.5):
        """Merge two weights, using the parameters s and t to interpolate between domain and dimension weights, respectively."""
        
        dom_weights = {}
        dim_weights = {}

        for dom in set(self.domain_weights.keys()) & set(other.domain_weights.keys()):
            weight_dom = s * self.domain_weights[dom] + (1.0 - s) * other.domain_weights[dom]
            dom_weights[dom] = weight_dom
            weights_dim = {}
            for dim in self.dimension_weights[dom].keys():
                w = t * self.dimension_weights[dom][dim] + (1.0 - t) * other.dimension_weights[dom][dim]
                weights_dim[dim] = w
            dim_weights[dom] = weights_dim
        
        for dom in set(self.domain_weights.keys()) - set(other.domain_weights.keys()):
            dom_weights[dom] = self.domain_weights[dom]
            dim_weights[dom] = self.dimension_weights[dom].copy()
        
        for dom in set(other.domain_weights.keys()) - set(self.domain_weights.keys()):
            dom_weights[dom] = other.domain_weights[dom]
            dim_weights[dom] = other.dimension_weights[dom].copy()
        
        
        return Weights(dom_weights, dim_weights)
    
    def project(self, new_domains):
        """Projects this set of weights onto a subset of domains."""

        dom_weights = {}
        dim_weights = {}
        for dom in new_domains.keys():
            dom_weights[dom] = self.domain_weights[dom]
            dim_weights[dom] = dict(self.dimension_weights[dom])
        
        return Weights(dom_weights, dim_weights)