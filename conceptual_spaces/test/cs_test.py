# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:32:16 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
from cs.cs import ConceptualSpace
from cs.weights import Weights
from math import sqrt

class TestCore(unittest.TestCase):

    # constructor()
    def test_constructor_fine(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        

        cs = ConceptualSpace(n, domains)        
        
        self.assertEqual(cs._n_dim, n)
        self.assertEqual(cs._domains, domains)
    
    def test_constructor_negative_n(self):
        n = -1        
        domains = {0:[0,1], 1:[2,3]}        
        with self.assertRaises(Exception):        
            ConceptualSpace(n, domains)
        
    def test_constructor_overlapping_domains(self):
        n = 4        
        domains = {0:[0,1,2], 1:[2,3]}        
        with self.assertRaises(Exception):        
            ConceptualSpace(n, domains)
    
    def test_constructor_illegal_dimensions(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3,4]}        
        with self.assertRaises(Exception):        
            ConceptualSpace(n, domains)
            
    def test_constructor_missing_dimensions(self):
        n = 4        
        domains = {0:[0,1], 1:[3]}        
        with self.assertRaises(Exception):        
            ConceptualSpace(n, domains)

    def test_constructor_empty_domain(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3], 2:[]}        
        with self.assertRaises(Exception):        
            ConceptualSpace(n, domains)

    # distance()
    def test_distance_illegal_point(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        cs = ConceptualSpace(n, domains)
        
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [5,6,7]
        with self.assertRaises(Exception):
            cs.distance(x,y,w)
    
    def test_distance_unit_diff_identically_weighted(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        cs = ConceptualSpace(n, domains)
        
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [2,3,2,3]   # distance of 1 wrt each coordinate
        self.assertEqual(cs.distance(x,y,w), 2.0)
        self.assertEqual(cs.distance(x,y,w), cs.distance(y,x,w))

    def test_distance_unit_diff_differently_weighted(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        cs = ConceptualSpace(n, domains)
        
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [2,3,2,3]   # distance of 1 wrt each coordinate
        self.assertEqual(cs.distance(x,y,w), 2.0)
        self.assertEqual(cs.distance(x,y,w), cs.distance(y,x,w))
        
    def test_distance_other_diff_identically_weighted(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        cs = ConceptualSpace(n, domains)
        
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [2,0,2,2]   # difference: 1 2 1 2
        self.assertEqual(cs.distance(x,y,w), sqrt(0.5*1 + 0.5*4) + sqrt(0.5*1 + 0.5*4))
        self.assertEqual(cs.distance(x,y,w), cs.distance(y,x,w))

    def test_distance_other_diff_differently_weighted(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        cs = ConceptualSpace(n, domains)
        
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [2,0,2,2]   # difference: 1 2 1 2
        self.assertEqual(cs.distance(x,y,w), (4.0/3)*sqrt(0.5*1+0.5*4) + (2.0/3)*sqrt(0.6*1 + 0.4*4))
        self.assertEqual(cs.distance(x,y,w), cs.distance(y,x,w))
    
unittest.main()