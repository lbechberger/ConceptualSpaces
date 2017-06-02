# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 13:51:31 2017

@author: lbechberger
"""

class Cuboid:
    """Single cuboid that is used to define a concept's core."""
    
    def __init__(self, p_min, p_max):
        """Initializes the cuboid.
        
        All entries of p_min must be <= their corresponding entry in p_max."""
        
        if not self._check(p_min, p_max):
            raise Exception("p_min is in some dimension above p_max")           
        
        self._p_min = p_min
        self._p_max = p_max
    
    def _check(self, p_min=None, p_max=None):
        """Asserts that no entry of _p_min is larger than the corresponding entry of _p_max."""
        
        p_min = p_min if (not p_min == None) else self._p_min
        p_max = p_max if (not p_max == None) else self._p_max
        
        if not len(p_min) == len(p_max):
            raise Exception("p_min and p_max are of different legnth")
        
        return reduce(lambda x, y: x and y, map(lambda y,z: y <= z, p_min, p_max))
    
    def contains(self, point):
        """Checks whether the given point is inside the cuboid."""
        
        if not len(self._p_min) == len(point):
            raise Exception("point has illegal dimensionality")
                
        return reduce(lambda x, y: x and y, map(lambda x,y,z: x <= y <= z, self._p_min, point, self._p_max))
        
    def _find_closest_point(self, point):
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

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self._p_min == other._p_min and self._p_max == other._p_max
        return False
    
    def __ne__(self, other):
        return not self.__eq__(other)

    def __str__(self):
        return "c({},{})".format(self._p_min, self._p_max)
    
    def intersect(self, other):
        """Intersects this cuboid with another one and returns the result as a new cuboid. Returns None if intersection is empty"""

        if not isinstance(other, self.__class__):
            raise Exception("can only intersect with other cuboids")
        
        if not len(other._p_min) == len(self._p_min):
            raise Exception("different dimensionality")

        p_min = []
        p_max = []

        for i in range(len(self._p_min)):
            if other._p_max[i] < self._p_min[i] or other._p_min[i] > self._p_max[i]:
                return None # no overlap in dimension i
            p_min.append(max(self._p_min[i], other._p_min[i]))
            p_max.append(min(self._p_max[i], other._p_max[i]))
        
        return Cuboid(p_min, p_max)