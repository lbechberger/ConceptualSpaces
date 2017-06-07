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
    
    # __eq__(), __ne__()
    def test_eq_ne_no_concept(self):
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10]), Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        f = Concept(s, 1.0, 2.0, w, cs)  
        self.assertFalse(f == 42)
        self.assertTrue(f != 42)
    
    def test_eq_ne_identity(self):
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10]), Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        f = Concept(s, 1.0, 2.0, w, cs)  
        self.assertTrue(f == f)
        self.assertFalse(f != f)

    def test_eq_ne_shallow_copy(self):
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10]), Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        f = Concept(s, 1.0, 2.0, w, cs)  
        f2 = Concept(s, 1.0, 2.0, w, cs)
        self.assertTrue(f == f2)
        self.assertFalse(f != f2)
    
    def test_eq_ne_other_params(self):
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10]), Cuboid([1,2,3,4],[3,4,5,6])])
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        f = Concept(s, 1.0, 2.0, w, cs)  
        f2 = Concept(s, 1.0, 3.0, w, cs)
        f3 = Concept(s, 0.5, 2.0, w, cs)
        self.assertFalse(f == f2)
        self.assertTrue(f != f2)
        self.assertFalse(f == f3)
        self.assertTrue(f != f3)
        self.assertFalse(f3 == f2)
        self.assertTrue(f3 != f2)
     
    # unify()
    def test_unify_no_repair_no_params(self):
        c1 = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([4,5,6],[7,7,7])
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(3, {0:[0], 1:[1,2]})
        f = Concept(Core([c1]), 1.0, 2.0, w, cs) 
        f2 = Concept(Core([c2]), 1.0, 2.0, w, cs)
        f_res = Concept(Core([c1,c2]), 1.0, 2.0, w, cs)
        self.assertEqual(f.unify(f2), f_res)
        self.assertEqual(f2.unify(f), f_res)

    def test_unify_no_repair_params(self):
        c1 = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([4,5,6],[7,7,7])
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        dim2 = {0:{0:1}, 1:{1:0.5, 2:0.5}}
        w2 = Weights(dom, dim2)
        dim_res = {0:{0:1}, 1:{1:0.55, 2:0.45}}
        w_res = Weights(dom, dim_res)
        cs = ConceptualSpace(3, {0:[0], 1:[1,2]})
        f = Concept(Core([c1]), 1.0, 5.2, w, cs) 
        f2 = Concept(Core([c2]), 0.5, 2.5, w2, cs)
        f_res = Concept(Core([c1,c2]), 1.0, 2.5, w_res, cs)
        self.assertEqual(f.unify(f2), f_res)
        self.assertEqual(f2.unify(f), f_res)
        
    def test_unify_repair_no_params(self):
        c1 = Cuboid([1,2,3],[2,3,4])
        c2 = Cuboid([3,4,5],[7,7,7])
        c1_res = Cuboid([1,2,3],[3.25,4,4.75])
        c2_res = Cuboid([3,4,4.75],[7,7,7])
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(3, {0:[0], 1:[1,2]})
        f = Concept(Core([c1]), 1.0, 2.0, w, cs) 
        f2 = Concept(Core([c2]), 1.0, 2.0, w, cs)
        f_res = Concept(Core([c1_res,c2_res]), 1.0, 2.0, w, cs)
        self.assertEqual(f.unify(f2), f_res)
        self.assertEqual(f2.unify(f), f_res)
        
#    def test_unify_identity(self):
#        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10]), Cuboid([1,2,3,4],[3,4,5,6])])
#        dom = {0:2, 1:1}        
#        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
#        w = Weights(dom, dim)
#        cs = ConceptualSpace(4, {0:[0,1], 1:[2,3]})
#        f = Concept(s, 1.0, 2.0, w, cs)  
#        self.assertEqual(f, f.unify(f))
    
    # cut()
    def test_cut_above(self):
        c1 = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([4,5,6],[7,7,7])
        s1 = Core([c1, c2])
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(3, {0:[0], 1:[1,2]})
        f1 = Concept(s1, 1.0, 2.0, w, cs)
        self.assertEqual(f1.cut(0,8.0), (f1, None))

    def test_cut_below(self):
        c1 = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([4,5,6],[7,7,7])
        s1 = Core([c1, c2])
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(3, {0:[0], 1:[1,2]})
        f1 = Concept(s1, 1.0, 2.0, w, cs)
        self.assertEqual(f1.cut(2,0.0), (None, f1))
        
    def test_cut_through_center(self):
        c1 = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([4,5,6],[7,7,7])
        s1 = Core([c1, c2])
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(3, {0:[0], 1:[1,2]})
        f1 = Concept(s1, 1.0, 2.0, w, cs)
        
        low_c1 = Cuboid([1,2,3],[5,8,9])
        low_c2 = Cuboid([4,5,6],[5,7,7])
        low_s = Core([low_c1, low_c2])
        low_f = Concept(low_s, 1.0, 2.0, w, cs)        
        
        up_c1 = Cuboid([5,2,3],[7,8,9])
        up_c2 = Cuboid([5,5,6],[7,7,7])
        up_s = Core([up_c1, up_c2])
        up_f = Concept(up_s, 1.0, 2.0, w, cs)
        
        self.assertEqual(f1.cut(0, 5), (low_f, up_f))

    def test_cut_through_one_cuboid(self):
        c1 = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([4,5,6],[7,7,7])
        s1 = Core([c1, c2])
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        cs = ConceptualSpace(3, {0:[0], 1:[1,2]})
        f1 = Concept(s1, 1.0, 2.0, w, cs)
        
        low_c1 = Cuboid([1,2,3],[7,8,5])
        low_s = Core([low_c1])
        low_f = Concept(low_s, 1.0, 2.0, w, cs)
        
        up_c1 = Cuboid([1,2,5],[7,8,9])
        up_c2 = Cuboid([4,5,6],[7,7,7])
        up_s = Core([up_c1, up_c2])
        up_f = Concept(up_s, 1.0, 2.0, w, cs)
        
        self.assertEqual(f1.cut(2, 5), (low_f, up_f))
    
unittest.main()