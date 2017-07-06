# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:50:58 2017

@author: lbechberger
"""
import cuboid as cub
import cs
from itertools import compress

class Core:
    """A concept's core, consisting of a set of cuboids with nonempty intersection.
    
    Implementation of the crisp Simple Star-Shaped Set (SSSS)"""
    
    def __init__(self, cuboids, domains):
        """Initializes the concept's core.
        
        The parameter cuboids must be a list of cuboids."""
        
        if not type(cuboids) == list:
            raise Exception("'cuboids' is not a list")           
        for c in cuboids:
            if not isinstance(c, cub.Cuboid):
                raise Exception("'cuboids' does not only contain cuboids")

        if len(cuboids) == 0:
            raise Exception("empty list of cuboids")

        if not check(cuboids, domains):
            raise Exception("cuboids do not intersect or are defined on different domains")
        
        self._cuboids = simplify(cuboids)
        self._domains = domains
    
    def add_cuboid(self, cuboid):
        """Adds the given cuboid to the internal list if it does not violate any constraints.
        
        Returns true if the addition was successful and false if it was not successful."""

        extended_list = list(self._cuboids) + [cuboid]
        if check(extended_list, self._domains):
            self._cuboids.append(cuboid)
            return True
        else:
            return False
    
    def __str__(self):
        return "{{{}}}".format(', '.join(str(x) for x in self._cuboids))
    
    def __eq__(self, other):
        if not isinstance(other, Core):
            return False
        if len(self._cuboids) != len(other._cuboids):
            return False
        for c in self._cuboids:
            if not c in other._cuboids:
                return False
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
        
    def find_closest_point_candidates(self, point):
        """Returns a list that contains for each cuboid the closest point in the cuboid to the given point."""
        
        return map(lambda c: c.find_closest_point(point), self._cuboids)
    
    def unify(self, other):
        """Computes the union of this core with another core."""
        
        if not isinstance(other, Core):
            raise Exception("Not a valid core")
        
        if not self._cuboids[0]._compatible(other._cuboids[0]):
            raise Exception("Incompatible cores")
        
        extended_list = list(self._cuboids) + list(other._cuboids)
        
        return from_cuboids(extended_list, self._domains)

    
    def cut(self, dimension, value):
        """Cuts the given core into two parts (at the given value on the given dimension).
        
        Returns the lower part and the upper part as a tuple (lower, upper)."""
        
        lower_cuboids = []
        upper_cuboids = []
        
        for cuboid in self._cuboids:
            if value >= cuboid._p_max[dimension]:
                lower_cuboids.append(cuboid)
            elif value <= cuboid._p_min[dimension]:
                upper_cuboids.append(cuboid)
            else:
                p_min = list(cuboid._p_min)
                p_min[dimension] = value
                p_max = list(cuboid._p_max)
                p_max[dimension] = value
                lower_cuboids.append(cub.Cuboid(list(cuboid._p_min), p_max, cuboid._domains))
                upper_cuboids.append(cub.Cuboid(p_min, list(cuboid._p_max), cuboid._domains))

        lower_core = None if len(lower_cuboids) == 0 else Core(lower_cuboids, self._domains)     
        upper_core = None if len(upper_cuboids) == 0 else Core(upper_cuboids, self._domains)     
        
        return lower_core, upper_core
    
    def project(self, new_domains):
        """Projects this core onto the given set of new domains (must be a subset of the core's current domains)."""
        
        if not all(dom in self._domains.items() for dom in new_domains.items()):
            raise Exception("Illegal set of new domains!")
        
        projected_cuboids = map(lambda c: c.project(new_domains), self._cuboids)
        
        return Core(projected_cuboids, new_domains)

    def midpoint(self):
        """Computes the midpoint of this core's central region."""
        central_region = self._cuboids[0]
        for c in self._cuboids:
            central_region = central_region.intersect(c)
        
        midpoint = map(lambda x, y: 0.5*(x + y), central_region._p_min, central_region._p_max)
        return midpoint

def check(cuboids, domains):
    """Asserts that the intersection of all cuboids is nonempty and that they are defined on the same domains"""

    intersection = cuboids[0]
    for c in cuboids:
        intersection = intersection.intersect(c)
        if intersection == None:
            return False

    if not all(dom in cs._domains.items() for dom in domains.items()):
        return False
    
    for c in cuboids:
        if c._domains != domains:
            return False
    
    return True

def from_cuboids(cuboids, domains):
    """Create a core from possibly non-intersecting cuboids by applying the repair mechanism."""
    
    if check(cuboids, domains):
        return Core(cuboids, domains)  # all cuboids already intersect --> nothing to do
    
    # need to perform repair mechanism        
    midpoints = []
    for cuboid in cuboids: # midpoint of each cuboid
        midpoints.append(map(lambda x, y: 0.5*(x + y), cuboid._p_min, cuboid._p_max))
    # sum up all midpoints & divide by number of cuboids
    midpoint = reduce(lambda x, y: map(lambda a,b: a+b, x, y), midpoints)
    midpoint = map(lambda x: x/len(cuboids), midpoint)
            
    # extend cuboids
    modified_cuboids = []
    for cuboid in cuboids:
        p_min = map(min, cuboid._p_min, midpoint)
        p_max = map(max, cuboid._p_max, midpoint)
        modified_cuboids.append(cub.Cuboid(p_min, p_max, cuboid._domains))
    
    return Core(modified_cuboids, domains)

def simplify(cuboids):
    """Simplifies the given set of cuboids by removing redundant ones."""
    
    keep = [True]*len(cuboids)
    for i in range(len(cuboids)):
        
        p_min = cuboids[i]._p_min
        p_max = cuboids[i]._p_max
        for j in range(len(cuboids)):
            if i == j or keep[j] == False:
                continue
            if cuboids[j].contains(p_min) and cuboids[j].contains(p_max):
                keep[i] = False
                break
        
    return list(compress(cuboids, keep))