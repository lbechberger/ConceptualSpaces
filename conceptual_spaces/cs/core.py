# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:50:58 2017

@author: lbechberger
"""
from cuboid import Cuboid

class Core:
    """A concept's core, consisting of a set of cuboids with nonempty intersection"""
    
    def __init__(self, cuboids):
        """Initializes the concept's core.
        
        The parameter cuboids must be a list of cuboids."""
        
        if not type(cuboids) == list:
            raise Exception("cuboids is not a list")           
        for c in cuboids:
            if not isinstance(c, Cuboid):
                raise Exception("cuboids does not only contain cuboids")

        if not self._check(cuboids):
            raise Exception("cuboids do not intersect")
        
        if len(cuboids) == 0:
            raise Exception("empty list of cuboids")
        
        self._cuboids = cuboids
    
    def _check(self, cuboids = None):
        """Asserts that the intersection of all cuboids is nonempty"""

        cuboids = cuboids if (not cuboids == None) else self._cuboids      
        
        intersection = cuboids[0]
        for c in cuboids:
            intersection = intersection.intersect(c)
            if intersection == None:
                return False
        
        return True
    
    def add_cuboid(self, cuboid):
        """Adds the given cuboid to the internal list if it does not violate any constraints.
        
        Returns true if the addition was successful and false if it was not successful."""

        extended_list = list(self._cuboids[:]) + [cuboid]
        if self._check(extended_list):
            self._cuboids.append(cuboid)
            return True
        else:
            return False
    