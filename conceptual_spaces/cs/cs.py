# -*- coding: utf-8 -*-
"""
The overall conceptual space.

Also provides some utility functions.

Created on Tue Jun  6 12:15:30 2017

@author: lbechberger
"""

from math import sqrt
import concept as con
from weights import Weights
import sys

this = sys.modules[__name__]

this._n_dim = None
this._domains = None
this._concepts = None
this._no_weights = None

def init(n_dim, domains):
    """Initializes a conceptual space with the given numer of dimensions and the given set of domains.
    
    'n_dim' is an integer >= 1 and 'domains' is a dictionary from domain ids to sets of dimensions."""

    if n_dim < 1:
        raise Exception("Need at least one dimension")
    
    if not _check_domain_structure(domains, n_dim):
        raise Exception("Invalid domain structure")
        
    this._n_dim = n_dim
    this._domains = domains
    this._concepts = {}
    
    # construct default weights
    dim_weights = {}
    dom_weights = {}
    for (dom, dims) in domains.items():
        dom_weights[dom] = 1.0
        local_dim_weights = {}
        for dim in dims:
            local_dim_weights[dim] = 1
        dim_weights[dom] = local_dim_weights
    this._no_weights = Weights(dom_weights, dim_weights)
    
def _check_domain_structure(domains, n_dim):
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

def distance(x, y, weights):
    """Computes the combined metric d_C(x,y,W) between the two points x and y using the weights in 'weights'."""
    
    if len(x) != this._n_dim or len(y) != this._n_dim:
        raise Exception("Points have wrong dimensionality")
    
    distance = 0.0

    for domain in this._domains.keys():
        inner_distance = 0.0
        if not domain in weights.domain_weights:    # don't take into account domains w/o weights
            continue
        for dimension in this._domains[domain]:
            inner_distance += weights.dimension_weights[domain][dimension] * (x[dimension] - y[dimension])**2
        distance += weights.domain_weights[domain] * sqrt(inner_distance)
    
    return distance

def add_concept(key, concept):
    """Adds a concept to the internal storage under the given key."""
    
    if not isinstance(concept, con.Concept):
        raise Exception("Not a valid concept")
    this._concepts[key] = concept

def delete_concept(key):
    """Deletes the concept with the given key form the internal storage."""
    
    if key in this._concepts:
        del this._concepts[key]

def between(first, middle, second, method="crisp"):
    """Computes the betweenness relation between the three given points.
    
    Right now only uses the crisp definition of betweenness (returns either 1.0 or 0.0)."""
    
    if method == "crisp":
        if (distance(first, middle, this._no_weights) + distance(middle, second, this._no_weights) - distance(first, second, this._no_weights)) < 0.00001:
            return 1.0
        else:
            return 0.0
    else:
        raise Exception("Unknown method")