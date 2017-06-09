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
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)        
        
        self.assertEqual(f._core, s)
        self.assertEqual(f._mu, 1.0)
        self.assertEqual(f._c, 2.0)
        self.assertEqual(f._weights, w)
    
    def test_constructor_wrong_core(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        with self.assertRaises(Exception):        
            Concept(42, 1.0, 2.0, w)        
        
    def test_constructor_wrong_mu(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 0.0, 2.0, w)        
    
    def test_constructor_wrong_c(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 1.0, -1.0, w)        
            
    def test_constructor_wrong_weigths(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 1.0, 1.0, 42)        

    def test_constructor_same_relevant_dimensions(self):
        ConceptualSpace(4, {0:[0], 1:[1,2,3]})
        c1 = Cuboid([float("-inf"),2,3,4],[float("inf"),5,6,7], {1:[1,2,3]})
        c2 = Cuboid([float("-inf"),3,4,5],[float("inf"),6,7,8], {1:[1,2,3]})
        s = Core([c1, c2], {1:[1,2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)        

        self.assertEqual(f._core, s)
        self.assertEqual(f._mu, 1.0)
        self.assertEqual(f._c, 2.0)
        self.assertEqual(f._weights, w)
    

    # membership()
    def test_membership_inside(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership(p), 1.0)
   
    def test_membership_inside_other_c(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 10.0, w)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership(p), 1.0)

    def test_membership_inside_other_mu(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 0.5, 2.0, w)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership(p), 0.5)
     
    def test_membership_outside_one_cuboid(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 4, 4]
        self.assertAlmostEqual(f.membership(p), 0.15173524)

    def test_membership_outside_two_cuboids(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 4, 4]
        self.assertAlmostEqual(f.membership(p), 0.15173524)
    
    def test_membership_inside_infinity(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([float("-inf"),float("-inf"),3,4],[float("inf"),float("inf"),5,6], {1:[2,3]})], {1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership(p), 1.0)
    
    def test_membership_outside_one_cuboid_infinity(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,float("-inf"),float("-inf")],[3,4,float("inf"),float("inf")], {0:[0,1]})], {0:[0,1]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 10, 4]
        self.assertAlmostEqual(f.membership(p), 0.15173524)
        
    def test_membership_outside_two_cuboids_infinity(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([float("-inf"), float("-inf"), 5, 6],[float("inf"), float("inf"), 10, 10], {1:[2,3]}), Cuboid([float("-inf"),float("-inf"),3,4],[float("inf"),float("inf"),5,6], {1:[2,3]})], {1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 10, 4]
        self.assertAlmostEqual(f.membership(p), 0.18515757)
    
    # __eq__(), __ne__()
    def test_eq_ne_no_concept(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        self.assertFalse(f == 42)
        self.assertTrue(f != 42)
    
    def test_eq_ne_identity(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        self.assertTrue(f == f)
        self.assertFalse(f != f)

    def test_eq_ne_shallow_copy(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        f2 = Concept(s, 1.0, 2.0, w)
        self.assertTrue(f == f2)
        self.assertFalse(f != f2)
    
    def test_eq_ne_other_params(self):
        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        f2 = Concept(s, 1.0, 3.0, w)
        f3 = Concept(s, 0.5, 2.0, w)
        self.assertFalse(f == f2)
        self.assertTrue(f != f2)
        self.assertFalse(f == f3)
        self.assertTrue(f != f3)
        self.assertFalse(f3 == f2)
        self.assertTrue(f3 != f2)
     
    # unify()
    def test_unify_no_repair_no_params(self):
        ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f = Concept(Core([c1], {0:[0], 1:[1,2]}), 1.0, 2.0, w) 
        f2 = Concept(Core([c2], {0:[0], 1:[1,2]}), 1.0, 2.0, w)
        f_res = Concept(Core([c1,c2], {0:[0], 1:[1,2]}), 1.0, 2.0, w)
        self.assertEqual(f.unify(f2), f_res)
        self.assertEqual(f2.unify(f), f_res)

    def test_unify_no_repair_params(self):
        ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        dim2 = {0:{0:1}, 1:{1:0.5, 2:0.5}}
        w2 = Weights(dom, dim2)
        dim_res = {0:{0:1}, 1:{1:0.55, 2:0.45}}
        w_res = Weights(dom, dim_res)
        f = Concept(Core([c1], {0:[0], 1:[1,2]}), 1.0, 5.2, w) 
        f2 = Concept(Core([c2], {0:[0], 1:[1,2]}), 0.5, 2.5, w2)
        f_res = Concept(Core([c1,c2], {0:[0], 1:[1,2]}), 1.0, 2.5, w_res)
        self.assertEqual(f.unify(f2), f_res)
        self.assertEqual(f2.unify(f), f_res)
        
    def test_unify_repair_no_params(self):
        ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[2,3,4], {0:[0], 1:[1,2]})
        c2 = Cuboid([3,4,5],[7,7,7], {0:[0], 1:[1,2]})
        c1_res = Cuboid([1,2,3],[3.25,4,4.75], {0:[0], 1:[1,2]})
        c2_res = Cuboid([3,4,4.75],[7,7,7], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f = Concept(Core([c1], {0:[0], 1:[1,2]}), 1.0, 2.0, w) 
        f2 = Concept(Core([c2], {0:[0], 1:[1,2]}), 1.0, 2.0, w)
        f_res = Concept(Core([c1_res,c2_res], {0:[0], 1:[1,2]}), 1.0, 2.0, w)
        self.assertEqual(f.unify(f2), f_res)
        self.assertEqual(f2.unify(f), f_res)
        
#    def test_unify_identity(self):
#        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10]), Cuboid([1,2,3,4],[3,4,5,6])])
#        dom = {0:2, 1:1}        
#        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
#        w = Weights(dom, dim)
#        ConceptualSpace(4, {0:[0,1], 1:[2,3]})
#        f = Concept(s, 1.0, 2.0, w)  
#        self.assertEqual(f, f.unify(f))
    
    # cut()
    def test_cut_above(self):
        ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        s1 = Core([c1, c2], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f1 = Concept(s1, 1.0, 2.0, w)
        self.assertEqual(f1.cut(0,8.0), (f1, None))

    def test_cut_below(self):
        ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        s1 = Core([c1, c2], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f1 = Concept(s1, 1.0, 2.0, w)
        self.assertEqual(f1.cut(2,0.0), (None, f1))
        
    def test_cut_through_center(self):
        ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        s1 = Core([c1, c2], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f1 = Concept(s1, 1.0, 2.0, w)
        
        low_c1 = Cuboid([1,2,3],[5,8,9], {0:[0], 1:[1,2]})
        low_c2 = Cuboid([4,5,6],[5,7,7], {0:[0], 1:[1,2]})
        low_s = Core([low_c1, low_c2], {0:[0], 1:[1,2]})
        low_f = Concept(low_s, 1.0, 2.0, w)        
        
        up_c1 = Cuboid([5,2,3],[7,8,9], {0:[0], 1:[1,2]})
        up_c2 = Cuboid([5,5,6],[7,7,7], {0:[0], 1:[1,2]})
        up_s = Core([up_c1, up_c2], {0:[0], 1:[1,2]})
        up_f = Concept(up_s, 1.0, 2.0, w)
        
        self.assertEqual(f1.cut(0, 5), (low_f, up_f))

    def test_cut_through_one_cuboid(self):
        ConceptualSpace(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        s1 = Core([c1, c2], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f1 = Concept(s1, 1.0, 2.0, w)
        
        low_c1 = Cuboid([1,2,3],[7,8,5], {0:[0], 1:[1,2]})
        low_s = Core([low_c1], {0:[0], 1:[1,2]})
        low_f = Concept(low_s, 1.0, 2.0, w)
        
        up_c1 = Cuboid([1,2,5],[7,8,9], {0:[0], 1:[1,2]})
        up_c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        up_s = Core([up_c1, up_c2], {0:[0], 1:[1,2]})
        up_f = Concept(up_s, 1.0, 2.0, w)
        
        self.assertEqual(f1.cut(2, 5), (low_f, up_f))
    
    # project()
    def test_project_identical_domains(self):
        ConceptualSpace(3, {0:[0,1,2]})
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        s = Core([c1],{0:[0,1,2]})
        w = Weights({0:1}, {0:{0:0.5, 1:0.3, 2:0.2}})
        f = Concept(s, 1.0, 5.0, w)
        self.assertEqual(f.project({0:[0,1,2]}), f)
    
    def test_project_correct(self):
        ConceptualSpace(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0,1], 1:[2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0,1], 1:[2]})
        s = Core([c1, c2],{0:[0,1], 1:[2]})
        w = Weights({0:1, 1:2}, {0:{0:0.5, 1:0.3}, 1:{2:1}})
        f = Concept(s, 1.0, 5.0, w)
        
        c1_res1 = Cuboid([1,2,float("-inf")],[7,8,float("inf")],{0:[0,1]})
        c2_res1 = Cuboid([4,5,float("-inf")],[7,7,float("inf")],{0:[0,1]})
        s_res1 = Core([c1_res1, c2_res1], {0:[0,1]})
        w_res1 = Weights({0:1}, {0:{0:0.5, 1:0.3}})
        f_res1 = Concept(s_res1, 1.0, 5.0, w_res1)
        
        c1_res2 = Cuboid([float("-inf"),float("-inf"),3],[float("inf"),float("inf"),9],{1:[2]})
        c2_res2 = Cuboid([float("-inf"),float("-inf"),6],[float("inf"),float("inf"),7],{1:[2]})
        s_res2 = Core([c1_res2, c2_res2], {1:[2]})
        w_res2 = Weights({1:1}, {1:{2:1}})
        f_res2 = Concept(s_res2, 1.0, 5, w_res2)
        
        self.assertEqual(f.project({0:[0,1]}), f_res1)
        self.assertEqual(f.project({1:[2]}), f_res2)

    
unittest.main()