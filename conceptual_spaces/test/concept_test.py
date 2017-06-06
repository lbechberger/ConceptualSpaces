# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:56:30 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
from cs.core import Core
from cs.cuboid import Cuboid
from cs.concept import Concept
from cs.weights import Weights
from cs.cs import ConceptualSpace

class TestCore(unittest.TestCase):

    # constructor()
    def test_constructor_fine(self):
        s = Core([Cuboid([1,2],[3,4])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        
        f = Concept(s, 1.0, 2.0, w, cs)        
        
        self.assertEqual(f._core, s)
        self.assertEqual(f._mu, 1.0)
        self.assertEqual(f._c, 2.0)
        self.assertEqual(f._weights, w)
    
    def test_constructor_wrong_core(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(42, 1.0, 2.0, w, cs)        
        
    def test_constructor_wrong_mu(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        s = Core([Cuboid([1,2],[3,4])])
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 0.0, 2.0, w, cs)        
    
    def test_constructor_wrong_c(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        s = Core([Cuboid([1,2],[3,4])])
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 1.0, -1.0, w, cs)        
            
    def test_constructor_wrong_weigths(self):
        s = Core([Cuboid([1,2],[3,4])])
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 1.0, 1.0, 42, cs)        

    # membership()
    def test_membership_inside(self):
        s = Core([Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        
        f = Concept(s, 1.0, 2.0, w, cs)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership(p), 1.0)
   
    def test_membership_inside_other_c(self):
        s = Core([Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        
        f = Concept(s, 1.0, 10.0, w, cs)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership(p), 1.0)

    def test_membership_inside_other_mu(self):
        s = Core([Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        
        f = Concept(s, 0.5, 2.0, w, cs)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership(p), 0.5)
     
    def test_membership_outside_one_cuboid(self):
        s = Core([Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        
        f = Concept(s, 1.0, 2.0, w, cs)  
        p = [4, 4, 4, 4]
        self.assertAlmostEqual(f.membership(p), 0.15173524)

    def test_membership_outside_two_cuboids(self):
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10]), Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        
        f = Concept(s, 1.0, 2.0, w, cs)  
        p = [4, 4, 4, 4]
        self.assertAlmostEqual(f.membership(p), 0.15173524)
    
unittest.main()