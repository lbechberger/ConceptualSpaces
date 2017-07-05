# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:45:36 2017

@author: lbechberger
"""

import cs

class Weights:
    """Specifies a set of weights W = <W_delta, {W_d}>.
    
    Weights must always fulfill their normalization constraints."""
    
    def __init__(self, domain_weights, dimension_weights):
        """Initializes the weights and normalizes them if necessary.
        
        'domain_weights' is a mapping from domains to weights and 
        'dimension_weights' contains for each domain a mapping from dimensions to weights."""
        
        self._domain_weights = self._normalize(domain_weights, len(domain_weights.keys()))
        self._dimension_weights = {}
        for (domain, weights) in dimension_weights.items():
            self._dimension_weights[domain] = self._normalize(weights, 1.0)
        
    def _normalize(self, weights, total):
        """Normalizes a given set of weights such that they sum up to the desired total."""
        
        result = {}
        old_sum = sum(weights.values())
        
        for (k,v) in weights.items():
            result[k] = (1.0*v*total)/(old_sum)
        
        return result

    def __str__(self):
        return "w<{0},<{1}>>".format(str(self._domain_weights),str(self._dimension_weights))
    
    def __eq__(self, other):
        if not isinstance(other, Weights):
            return False
            
        if len(self._domain_weights) != len(other._domain_weights):
            return False
        for dom, weight in self._domain_weights.iteritems():
            if dom not in other._domain_weights:
                return False
            if not cs.equal(other._domain_weights[dom], weight):
                return False
        
        if len(self._dimension_weights) != len(other._dimension_weights):
            return False
        for dom, dims in self._dimension_weights.iteritems():
            if dom not in other._dimension_weights:
                return False
            other_dims = other._dimension_weights[dom]
            if len(dims) != len(other_dims):
                return False
            for dim, weight in dims.iteritems():
                if dim not in other_dims:
                    return False
                if not cs.equal(other_dims[dim], weight):
                    return False
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def merge(self, other, s = 0.5, t = 0.5):
        """Merge two weights, using the parameters s and t to interpolate between domain and dimension weights, respectively."""
        
        dom_weights = {}
        dim_weights = {}

        for dom in set(self._domain_weights.keys()) & set(other._domain_weights.keys()):
            weight_dom = s * self._domain_weights[dom] + (1.0 - s) * other._domain_weights[dom]
            dom_weights[dom] = weight_dom
            weights_dim = {}
            for dim in self._dimension_weights[dom].keys():
                w = t * self._dimension_weights[dom][dim] + (1.0 - t) * other._dimension_weights[dom][dim]
                weights_dim[dim] = w
            dim_weights[dom] = weights_dim
        
        for dom in set(self._domain_weights.keys()) - set(other._domain_weights.keys()):
            dom_weights[dom] = self._domain_weights[dom]
            dim_weights[dom] = self._dimension_weights[dom].copy()
        
        for dom in set(other._domain_weights.keys()) - set(self._domain_weights.keys()):
            dom_weights[dom] = other._domain_weights[dom]
            dim_weights[dom] = other._dimension_weights[dom].copy()
        
        
        return Weights(dom_weights, dim_weights)
    
    def project(self, new_domains):
        """Projects this set of weights onto a subset of domains."""

        dom_weights = {}
        dim_weights = {}
        for dom in new_domains.keys():
            dom_weights[dom] = self._domain_weights[dom]
            dim_weights[dom] = dict(self._dimension_weights[dom])
        
        return Weights(dom_weights, dim_weights)

def check(domain_weights, dimension_weights):
    """Checks if all normalization constraints are fulfilled."""

    if not cs.equal(sum(domain_weights.values()), len(domain_weights.keys())):
        return False
    
    for weights in dimension_weights.values():
        if not cs.equal(sum(weights.values()), 1.0):
            return False
    
    return True
