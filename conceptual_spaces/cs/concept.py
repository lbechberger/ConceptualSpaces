# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: lbechberger
"""

from math import exp, sqrt, factorial, pi, gamma
import itertools

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

    def _reduce_domains(self, domains, dimensions):
        """Reduces the domain structure such that only the given dimensions are still contained."""
        new_domains = {}

        for (dom, dims) in domains.items():
            filtered_dims = [dim for dim in set(dims) & set(dimensions)]
            if len(filtered_dims) > 0:
                new_domains[dom] = filtered_dims
        
        return new_domains

    def _hypervolume_couboid(self, cuboid):
        """Computes the hypervolume of a single fuzzified cuboid."""

        n = cs.ConceptualSpace.cs._n_dim

        # calculating the factor in front of the sum
        weight_product = 1.0
        for (dom, dom_weight) in self._weights.domain_weights.items():
            for (dim, dim_weight) in self._weights.dimension_weights[dom].items():
                weight_product *= dom_weight * sqrt(dim_weight)
        factor = self._mu / (self._c**n * weight_product)

        all_dims = [dim for domain in self._core._domains.values() for dim in domain]
        outer_sum = 0.0        
        # outer sum
        for i in range(1, n+1):
            subsets = list(itertools.combinations(all_dims, i))
            inner_sum = 0.0
            # inner sum
            for subset in subsets:
                # first product
                first_product = 1.0
                for dim in set(all_dims) - set(subset):
                    dom = filter(lambda (x,y): dim in y, self._core._domains.items())[0][0]
                    w_dom = self._weights.domain_weights[dom]
                    w_dim = self._weights.dimension_weights[dom][dim]
                    b = cuboid._p_max[dim] - cuboid._p_min[dim]
                    first_product *= w_dom * sqrt(w_dim) * b * self._c
                
                # second product
                second_product = 1.0
                reduced_domain_structure = self._reduce_domains(self._core._domains, subset)
                for (dom, dims) in reduced_domain_structure.items():
                    n_domain = len(dims)
                    second_product *= factorial(n_domain) * (pi ** (n_domain/2.0))/(gamma((n_domain/2.0) + 1))
                
                inner_sum += first_product * second_product
            
            outer_sum += inner_sum
        return factor * outer_sum

    def hypervolume(self):
        """Computes the hypervolume of this concept."""
        
        hypervolume = 0.0
        num_cuboids = len(self._core._cuboids)
        
        # use the inclusion-exclusion formula over all the cuboids
        for l in range(1, num_cuboids + 1):
            inner_sum = 0.0

            subsets = list(itertools.combinations(self._core._cuboids, l))           
            for subset in subsets:
                intersection = subset[0]
                for cuboid in subset:
                    intersection = intersection.intersect(cuboid)
                inner_sum += self._hypervolume_couboid(intersection)
                
            hypervolume += inner_sum * (-1.0)**(l+1)
        
        return hypervolume

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