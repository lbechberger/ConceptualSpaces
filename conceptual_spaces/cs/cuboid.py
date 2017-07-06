# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:51:31 2017

@author: lbechberger
"""

import cs

class Cuboid:
    """Single cuboid that is used to define a concept's core."""
    
    def __init__(self, p_min, p_max, domains):
        """Initializes the cuboid.
        
        All entries of p_min must be <= their corresponding entry in p_max.
        All dimensions contained in the domains must be finite, all other dimensions infinite."""
        
        if not check(p_min, p_max, domains):
            raise Exception("some constraint is violated!")           
        
        self._p_min = p_min
        self._p_max = p_max
        self._domains = domains        
  
    def contains(self, point):
        """Checks whether the given point is inside the cuboid."""
        
        if not len(self._p_min) == len(point):
            raise Exception("point has illegal dimensionality")
                
        return reduce(lambda x, y: x and y, map(lambda x,y,z: x <= y <= z, self._p_min, point, self._p_max))
        
    def find_closest_point(self, point):
        """Finds the point in the cuboid that is closest to the given point"""
        
        if not len(self._p_min) == len(point):
            raise Exception("point has illegal dimensionality")
        
        def helper(x,y,z):
            if x <= y <= z:
                return y
            elif x > y:
                return x
            else:
                return z
        
        return map(helper, self._p_min, point, self._p_max)

    def get_closest_points(self, other):
        """Computes closest points a in this and b in the other cuboid."""
        
        if not self._compatible(other):
            raise Exception("Cuboids not compatible")
        
        a = []
        b = []
        
        for i in range(len(self._p_min)):
            if other._p_max[i] < self._p_min[i]:    # other cuboid below this one
                a.append([self._p_min[i], self._p_min[i]])
                b.append([other._p_max[i], other._p_max[i]])
            elif other._p_min[i] > self._p_max[i]:  # this cuboid below other one
                a.append([self._p_max[i], self._p_max[i]])
                b.append([other._p_min[i], other._p_min[i]])
            else:                                   # cuboids intersect
                left = max(self._p_min[i], other._p_min[i])
                right = min(self._p_max[i], other._p_max[i])
                a.append([left, right])
                b.append([left, right])

        return a, b

    def __eq__(self, other):
        if isinstance(other, Cuboid):
            p_min_equal = reduce(lambda x,y: x and y, map(cs.equal, self._p_min, other._p_min))
            p_max_equal = reduce(lambda x,y: x and y, map(cs.equal, self._p_max, other._p_max))
            return p_min_equal and p_max_equal and self._domains == other._domains
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "{}-{}".format(self._p_min, self._p_max)
    
    def _compatible(self, other):
        """Checks whether two cuboids are compatible with each other (i.e., defined on compatible domain structures)."""

        if not isinstance(other, Cuboid):
            return False
        
        if len(self._p_min) != len(other._p_min):
            return False
        
        for dom in set(self._domains.keys()) & set(other._domains.keys()):
            if self._domains[dom] != other._domains[dom]:
                return False
        
        dom_union = dict(self._domains)
        dom_union.update(other._domains)
        return all(dom in cs._domains.items() for dom in dom_union.items())
    
    def intersect(self, other):
        """Intersects this cuboid with another one and returns the result as a new cuboid. Returns None if intersection is empty"""

        if not self._compatible(other):
            raise Exception("Cuboids are not compatible")

        p_min = []
        p_max = []

        for i in range(len(self._p_min)):
            if other._p_max[i] < self._p_min[i] or other._p_min[i] > self._p_max[i]:
                return None # no overlap in dimension i
            p_min.append(max(self._p_min[i], other._p_min[i]))
            p_max.append(min(self._p_max[i], other._p_max[i]))

        dom_union = dict(self._domains)
        dom_union.update(other._domains)         
        
        return Cuboid(p_min, p_max, dom_union)
        
    def project(self, new_domains):
        """Projects this cuboid onto the given domains (which must be a subset of the cuboid's current domains)."""
        
        if not all(dom in self._domains.items() for dom in new_domains.items()):
            raise Exception("Illegal set of new domains!")
        
        # remove all domains that became irrelevant by replacing the p_min and p_max entries with -inf and inf, respectively
        relevant_dims = [dim for domain in new_domains.values() for dim in domain]
        p_min = []
        p_max = []
        for i in range(len(self._p_min)):
            if i in relevant_dims:
                p_min.append(self._p_min[i])
                p_max.append(self._p_max[i])
            else:
                p_min.append(float("-inf"))
                p_max.append(float("inf"))
        
        return Cuboid(p_min, p_max, new_domains)

def check(p_min, p_max, domains):
    """Asserts that no entry of _p_min is larger than the corresponding entry of _p_max, 
    that both are defined on the correct set of dimensions, 
    and that the given domains are compatible with the overall domain structure of the conceptual space."""
    
    dims = [dim for domain in domains.values() for dim in domain]        
    
    if not len(p_min) == len(p_max) == cs._n_dim:
        return False

    for i in range(len(p_max)):
        if i in dims and (p_max[i] == float("inf") or p_min[i] == float("-inf")):
            return False
        if i not in dims and (p_max[i] != float("inf") or p_min[i] != float("-inf")):
            return False

    if not all(dom in cs._domains.items() for dom in domains.items()):
        return False
    
    return reduce(lambda x, y: x and y, map(lambda y,z: y <= z, p_min, p_max))
  