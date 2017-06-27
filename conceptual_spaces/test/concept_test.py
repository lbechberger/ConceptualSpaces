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

    # hypervolume()
    def test_hypervolume_single_cuboid_lemon(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_lemon = Cuboid([0.7, 0.45, 0.0], [0.8, 0.55, 0.1], domains)
        s_lemon = Core([c_lemon], domains)
        w_lemon = Weights({"color":0.5, "shape":0.5, "taste":2.0}, w_dim)
        f_lemon = Concept(s_lemon, 1.0, 20.0, w_lemon)
        
        self.assertAlmostEqual(f_lemon.hypervolume(), 54.0/4000.0)

    def test_hypervolume_single_cuboid_granny_smith(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
        s_granny_smith = Core([c_granny_smith], domains)
        w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_granny_smith = Concept(s_granny_smith, 1.0, 25.0, w_granny_smith)

        self.assertAlmostEqual(f_granny_smith.hypervolume(), 0.004212)

    def test_hypervolume_single_cuboid_pear(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        f_pear = Concept(s_pear, 1.0, 10.0, w_pear)

        self.assertAlmostEqual(f_pear.hypervolume(), 0.0561600)
 
    def test_hypervolume_single_cuboid_orange(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_orange = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
        s_orange = Core([c_orange], domains)
        w_orange = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_orange = Concept(s_orange, 1.0, 15.0, w_orange)

        self.assertAlmostEqual(f_orange.hypervolume(), 0.01270370)
   
    def test_hypervolume_multiple_cuboids_apple(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        f_apple = Concept(s_apple, 1.0, 5.0, w_apple)

        self.assertAlmostEqual(f_apple.hypervolume(), 0.3375000)
    
    # intersect()
    # coding: {num_cuboids}_{space_dim}_{space_type}_{intersection_type}_{mu}_{weights}_{c}_{alpha}
    def test_intersect_1C_2D_M_crisp_sameMu_sameW_sameC(self):
        doms = {0:[0], 1:[1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([1,2],[7,8], doms)
        c2 = Cuboid([4,5],[7,7], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1, 1:2}, {0:{0:1}, 1:{1:1}})
        f1 = Concept(s1, 1.0, 5.0, w)
        f2 = Concept(s2, 1.0, 5.0, w)
        
        c_res = Cuboid([4,5],[7,7], doms)
        s_res = Core([c_res], doms)
        f_res = Concept(s_res, 1.0, 5.0, w)
        
        self.assertEqual(f1.intersect(f2), f_res)
        self.assertEqual(f2.intersect(f1), f_res)

    def test_intersect_1C_2D_E_crisp_sameMu_sameW_sameC(self):
        doms = {0:[0,1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([4,2],[7,8], doms)
        c2 = Cuboid([6,5],[9,7], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1}, {0:{0:2, 1:3}})
        f1 = Concept(s1, 0.5, 7.0, w)
        f2 = Concept(s2, 0.5, 7.0, w)
        
        c_res = Cuboid([6,5],[7,7], doms)
        s_res = Core([c_res], doms)
        f_res = Concept(s_res, 0.5, 7.0, w)
        
        self.assertEqual(f1.intersect(f2), f_res)
        self.assertEqual(f2.intersect(f1), f_res)

    def test_intersect_1C_2D_E_crisp_diffMu_diffW_diffC(self):
        doms = {0:[0,1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([-0.25,-10],[0.45,-4], doms)
        c2 = Cuboid([0.45,-5],[0.7,42], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1}, {0:{0:2, 1:3}})
        w2 = Weights({0:1}, {0:{0:1, 1:1}})
        f1 = Concept(s1, 1.0, 12.0, w1)
        f2 = Concept(s2, 0.8, 3.0, w2)
        
        c_res = Cuboid([0.45,-5],[0.45,-4], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:1}, {0:{0:0.45, 1:0.55}})
        f_res = Concept(s_res, 0.8, 3.0, w_res)
        
        self.assertEqual(f1.intersect(f2), f_res)
        self.assertEqual(f2.intersect(f1), f_res)
 
    def test_intersect_1C_2D_E_muOverlap_diffMu_sameW_sameC(self):
        doms = {0:[0,1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([0.00,0.00],[0.25,0.25], doms)
        c2 = Cuboid([0.50,0.50],[0.75,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1}, {0:{0:1, 1:1}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 0.5, 1.0, w)

        c_res = Cuboid([0.5,0.50],[0.75,1.00], doms)
        s_res = Core([c_res], doms)
        f_res = Concept(s_res, 0.5, 1.0, w)
        
        self.assertEqual(f1.intersect(f2), f_res)
        self.assertEqual(f2.intersect(f1), f_res)

    def test_intersect_1C_2D_E_muOverlap_diffMu_sameW_sameC_variant2(self):
        doms = {0:[0,1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([0.00,0.00],[0.25,0.25], doms)
        c2 = Cuboid([0.50,0.50],[0.75,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1}, {0:{0:1, 1:1}})
        f1 = Concept(s1, 1.0, 2.0, w)
        f2 = Concept(s2, 0.5, 2.0, w)

        c_res = Cuboid([0.5,0.50],[0.6715762226,0.6715762226], doms)
        s_res = Core([c_res], doms)
        f_res = Concept(s_res, 0.5, 2.0, w)
        
        self.assertEqual(f1.intersect(f2), f_res)
        self.assertEqual(f2.intersect(f1), f_res)

    def test_intersect_1C_2D_E_1diffPoints_sameMu_sameW_sameC(self):
        doms = {0:[0,1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([0.00,0.00],[0.25,0.25], doms)
        c2 = Cuboid([0.25,0.50],[0.75,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1}, {0:{0:1, 1:1}})
        f1 = Concept(s1, 1.0, 5.0, w)
        f2 = Concept(s2, 1.0, 5.0, w)

        c_res = Cuboid([0.25,0.375],[0.25,0.375], doms)
        s_res = Core([c_res], doms)
        f_res = Concept(s_res, 0.6427870843, 5.0, w)
        
        self.assertEqual(f1.intersect(f2), f_res)
        self.assertEqual(f2.intersect(f1), f_res)

    def test_intersect_1C_2D_E_1diffExtrude_sameMu_sameW_sameC(self):
        doms = {0:[0,1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([0.00,0.00],[0.30,0.25], doms)
        c2 = Cuboid([0.20,0.50],[0.75,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1}, {0:{0:1, 1:1}})
        f1 = Concept(s1, 1.0, 5.0, w)
        f2 = Concept(s2, 1.0, 5.0, w)

        c_res = Cuboid([0.2,0.375],[0.30,0.375], doms)
        s_res = Core([c_res], doms)
        f_res = Concept(s_res, 0.6427870843, 5.0, w)
        
        self.assertEqual(f1.intersect(f2), f_res)
        self.assertEqual(f2.intersect(f1), f_res)

    def test_intersect_1C_2D_E_2diff_diffMu_diffW_diffC(self):
        doms = {0:[0,1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([0.00,0.00],[0.30,0.25], doms)
        c2 = Cuboid([0.40,0.50],[0.55,0.90], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1}, {0:{0:1, 1:1}})
        w2 = Weights({0:1}, {0:{0:2, 1:1}})
        f1 = Concept(s1, 0.9, 5.0, w1)
        f2 = Concept(s2, 1.0, 8.0, w2)

        c_res1 = Cuboid([0.3671005749,0.3762216484],[0.3671005749,0.3762216484], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1}, {0:{0:(7/12.0), 1:(5/12.0)}})        
        f_res1 = Concept(s_res1, 0.542937404, 5.0, w_res1)
        
        c_res2 = Cuboid([0.3670016362, 0.3762744145],[0.3670016362, 0.3762744145], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1}, {0:{0:(7/12.0), 1:(5/12.0)}})        
        f_res2 = Concept(s_res2, 0.5429368707, 5.0, w_res2)
        
        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_1C_2D_M_2diff_diffMu_diffW_diffC(self):
        doms = {0:[0],1:[1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([0.00,0.00],[0.30,0.25], doms)
        c2 = Cuboid([0.40,0.50],[0.55,0.90], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})
        w2 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})
        f1 = Concept(s1, 0.9, 5.0, w1)
        f2 = Concept(s2, 1.0, 8.0, w2)

        c_res1 = Cuboid([0.3999999999, 0.3204489823],[0.3999999999, 0.3204489823], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:(7/12.0), 1:(5/12.0)}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.3838108497, 5.0, w_res1)
        
        c_res2 = Cuboid([0.3999999999, 0.3204489825],[0.3999999999, 0.3204489825], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:(7/12.0), 1:(5/12.0)}, {0:{0:1}, 1:{1:1}})        
        f_res2 = Concept(s_res2, 0.3838108493, 5.0, w_res2)
              
        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_1C_2D_M_2diff_diffMu_sameW_diffC(self):
        doms = {0:[0],1:[1]}       
        ConceptualSpace(2, doms)
        c1 = Cuboid([0.00,0.00],[0.30,0.25], doms)
        c2 = Cuboid([0.40,0.50],[0.55,0.90], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})
        w2 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})
        f1 = Concept(s1, 0.9, 5.0, w1)
        f2 = Concept(s2, 1.0, 8.0, w2)

        c_res1 = Cuboid([0.3073830473, 0.3147660942],[0.4, 0.5], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.3723525481, 5.0, w_res1)
        
        c_res2 = Cuboid([0.3073830471, 0.3147660944],[0.4, 0.5], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res2 = Concept(s_res2, 0.3723525481, 5.0, w_res2)
              
        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_2C_2D_M_2diff_sameMu_sameW_sameC_sameAlpha(self):
        doms = {0:[0],1:[1]}       
        ConceptualSpace(2, doms)
        c11 = Cuboid([0.00,0.00],[0.20,0.70], doms)
        c12 = Cuboid([0.00,0.00],[0.60,0.10], doms)
        c21 = Cuboid([0.30,0.90],[1.00,1.00], doms)
        c22 = Cuboid([0.70,0.30],[1.00,1.00], doms)
        s1 = Core([c11, c12], doms)
        s2 = Core([c21, c22], doms)
        w1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})
        f1 = Concept(s1, 0.8, 10.0, w1)
        f2 = Concept(s2, 0.8, 10.0, w2)

        c_res11 = Cuboid([0.20, 0.50],[0.45, 0.85], doms)
        c_res12 = Cuboid([0.45, 0.15],[0.70, 0.50], doms)
        s_res1 = Core([c_res11, c_res12], doms)
        w_res1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.1785041281, 10.0, w_res1)
        
        c_res21 = Cuboid([0.20, 0.50],[0.45, 0.85], doms)
        c_res22 = Cuboid([0.45, 0.15],[0.70, 0.50], doms)
        s_res2 = Core([c_res21, c_res22], doms)
        w_res2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res2 = Concept(s_res2, 0.1785041281, 10.0, w_res2)
        
        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_2C_2D_M_2diff_sameMu_sameW_sameC_diffAlpha(self):
        doms = {0:[0],1:[1]}       
        ConceptualSpace(2, doms)
        c11 = Cuboid([0.00,0.00],[0.20,0.70], doms)
        c12 = Cuboid([0.00,0.00],[0.60,0.10], doms)
        c21 = Cuboid([0.50,0.90],[1.00,1.00], doms)
        c22 = Cuboid([0.70,0.30],[1.00,1.00], doms)
        s1 = Core([c11, c12], doms)
        s2 = Core([c21, c22], doms)
        w1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})
        f1 = Concept(s1, 0.8, 10.0, w1)
        f2 = Concept(s2, 0.8, 10.0, w2)

        c_res1 = Cuboid([0.60, 0.15],[0.70, 0.25], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.1785041281, 10.0, w_res1)
        
        c_res2 = Cuboid([0.60, 0.15],[0.70, 0.25], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res2 = Concept(s_res2, 0.1785041281, 10.0, w_res2)
        
        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_1C_3D_C_2diffExtrM_sameMu_depW_sameC(self):
        doms = {0:[0,1],1:[2]}       
        ConceptualSpace(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.40], doms)
        c2 = Cuboid([0.50,0.90,0.30],[1.00,1.00,0.70], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1, 1:2}, {0:{0:3, 1:1}, 1:{2:1}})
        w2 = Weights({0:1, 1:2}, {0:{0:1, 1:1}, 1:{2:1}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)

        c_res1 = Cuboid([0.3272802977, 0.837442891, 0.3],[0.3272802977, 0.837442891, 0.4], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:(2.0/3), 1:(4.0/3)}, {0:{0:0.625, 1:0.375}, 1:{2:1}})
        f_res1 = Concept(s_res1, 0.8409744422, 2.0, w_res1)
        
        c_res2 = Cuboid([0.3271963787, 0.8376746001, 0.3],[0.3271963787, 0.8376746001, 0.4], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1, 1:2}, {0:{0:0.625, 1:0.375}, 1:{2:1}})
        f_res2 = Concept(s_res2, 0.8409747409, 2.0, w_res2)
        
        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_1C_3D_C_2diffExtrE_sameMu_depW_sameC(self):
        doms = {0:[0],1:[1,2]}       
        ConceptualSpace(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.40], doms)
        c2 = Cuboid([0.50,0.90,0.30],[1.00,1.00,0.70], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:(2/3.0), 1:(4/3.0)}, {0:{0:1}, 1:{1:0.125, 2:0.875}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)

        c_res1 = Cuboid([0.3234316027, 0.7, 0.3],[0.4648527395, 0.9, 0.4], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:(5/6.0), 1:(7/6.0)}, {0:{0:1}, 1:{1:0.3125, 2:0.6875}})
        f_res1 = Concept(s_res1, 0.702480783, 2.0, w_res1)
        
        c_res2 = Cuboid([0.3234312906, 0.7, 0.3],[0.464853139, 0.9, 0.4], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:(5/6.0), 1:(7/6.0)}, {0:{0:1}, 1:{1:0.3125, 2:0.6875}})
        f_res2 = Concept(s_res2, 0.7024810437, 2.0, w_res2)
         
        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_1C_3D_C_3diff_sameMu_sameW_sameC(self):
        doms = {0:[0],1:[1,2]}       
        ConceptualSpace(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.30], doms)
        c2 = Cuboid([0.50,0.90,0.40],[1.00,1.00,0.70], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        w2 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)

        c_res1 = Cuboid([0.3279520734, 0.7, 0.3],[0.3720479266, 0.9, 0.4], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        f_res1 = Concept(s_res1, 0.5968175744, 2.0, w_res1)
        
        c_res2 = Cuboid([0.327952071, 0.7, 0.3],[0.372047929, 0.9, 0.4], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        f_res2 = Concept(s_res2, 0.5968175744, 2.0, w_res2)

        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_1C_3D_C_3diff_sameMu_diffW_sameC(self):
        doms = {0:[0],1:[1,2]}       
        ConceptualSpace(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.30], doms)
        c2 = Cuboid([0.50,0.90,0.40],[1.00,1.00,0.70], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.6, 2:0.4}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)

        c_res1 = Cuboid([0.2935424869, 0.8999999962, 0.4],[0.2935424869, 0.8999999962, 0.4], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1.25, 1:0.75}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res1 = Concept(s_res1, 0.6617185102, 2.0, w_res1)
        
        c_res2 = Cuboid([0.2935424944, 0.8999999007, 0.4],[0.2935424944, 0.8999999007, 0.4], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1.25, 1:0.75}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res2 = Concept(s_res2, 0.6617184173, 2.0, w_res2)

        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_2C_3D_C_3diffMuOverlap_diffMu_diffW_diffC_diffAlpha(self):
        doms = {0:[0],1:[1,2]}       
        ConceptualSpace(3, doms)
        c11 = Cuboid([0.00,0.00,0.00],[0.85,0.20,0.20], doms)
        c12 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.80], doms)
        c21 = Cuboid([0.90,0.30,0.30],[1.00,1.00,1.00], doms)
        c22 = Cuboid([0.40,0.40,0.90],[1.00,1.00,1.00], doms)
        s1 = Core([c11, c12], doms)
        s2 = Core([c21, c22], doms)
        w1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.6, 2:0.4}})
        f1 = Concept(s1, 1.0, 5.0, w1)
        f2 = Concept(s2, 0.5, 6.0, w2)

        c_res1 = Cuboid([0.90, 0.30, 0.30],[0.9090862907, 0.3864923883, 0.3351286822], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1.25, 1:0.75}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res1 = Concept(s_res1, 0.5, 5.0, w_res1)
        
        c_res2 = Cuboid([0.90, 0.30, 0.30],[0.9090862907, 0.3864923883, 0.3351286822], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1.25, 1:0.75}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res2 = Concept(s_res2, 0.5, 5.0, w_res2)

        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_1C_3D_C_3diffExtr_sameMu_diffW_diffC_2D3Dcuboids(self):
        doms = {0:[0],1:[1,2]}       
        ConceptualSpace(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.30], doms)
        c2 = Cuboid([-float("inf"),0.90,0.35],[float("inf"),1.00,0.70], {1:[1,2]})
        s1 = Core([c1], doms)
        s2 = Core([c2],  {1:[1,2]})
        w1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        w2 = Weights({1:1}, {1:{1:0.6, 2:0.4}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 5.0, w2)

        c_res1 = Cuboid([0.00, 0.8798325911, 0.3332302073],[0.20, 0.8798325911, 0.3332302073], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:(4.0/3.0), 1:(2.0/3.0)}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res1 = Concept(s_res1, 0.909910215, 2.0, w_res1)
        
        c_res2 = Cuboid([0.00, 0.8798294109, 0.3332359837],[0.20, 0.8798294109, 0.3332359837], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:(4.0/3.0), 1:(2.0/3.0)}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res2 = Concept(s_res2, 0.9099102768, 2.0, w_res2)

        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_1C_3D_C_3diffExtr_sameMu_diffW_diffC_2D1Dcuboids(self):
        doms = {0:[0],1:[1,2]}       
        ConceptualSpace(3, doms)
        c1 = Cuboid([0.00,-float("inf"),-float("inf")],[0.20,float("inf"),float("inf")], {0:[0]})
        c2 = Cuboid([-float("inf"),0.90,0.35],[float("inf"),1.00,0.70], {1:[1,2]})
        s1 = Core([c1], {0:[0]})
        s2 = Core([c2],  {1:[1,2]})
        w1 = Weights({0:1}, {0:{0:1}})
        w2 = Weights({1:1}, {1:{1:0.6, 2:0.4}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 5.0, w2)

        c_res1 = Cuboid([0.00, 0.90, 0.35],[0.20, 1.00, 0.70], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.6, 2:0.4}})
        f_res1 = Concept(s_res1, 1.00, 2.0, w_res1)
        
        c_res2 = Cuboid([0.00, 0.90, 0.35],[0.20, 1.00, 0.70], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.6, 2:0.4}})
        f_res2 = Concept(s_res2, 1.00, 2.0, w_res2)

        self.assertEqual(f1.intersect(f2), f_res1)
        self.assertEqual(f2.intersect(f1), f_res2)

    def test_intersect_apple_pear(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        f_pear = Concept(s_pear, 1.0, 10.0, w_pear)
        
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        f_apple = Concept(s_apple, 1.0, 5.0, w_apple)

        c_res1 = Cuboid([0.50, 0.6187499793, 0.35],[0.70, 0.6187499793, 0.45], domains)
        s_res1 = Core([c_res1], domains)
        w_res1 = Weights({"color":0.50, "shape":1.375, "taste":1.125}, w_dim)
        f_res1 = Concept(s_res1, 0.7910649883, 5.0, w_res1)
        
        c_res2 = Cuboid([0.50, 0.6187499794, 0.35],[0.70, 0.6187499794, 0.45], domains)
        s_res2 = Core([c_res2], domains)
        w_res2 = Weights({"color":0.50, "shape":1.375, "taste":1.125}, w_dim)
        f_res2 = Concept(s_res2, 0.791065315, 5.0, w_res2)
        
        self.assertEqual(f_apple.intersect(f_pear), f_res1)
        self.assertEqual(f_pear.intersect(f_apple), f_res2)

    # subset_of()
    def test_subset_of_granny_smith_apple(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
        s_granny_smith = Core([c_granny_smith], domains)
        w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_granny_smith = Concept(s_granny_smith, 1.0, 25.0, w_granny_smith)
        
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        f_apple = Concept(s_apple, 1.0, 5.0, w_apple)
        
        self.assertAlmostEqual(f_granny_smith.subset_of(f_apple), 1.00)

    def test_subset_of_pear_apple(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        f_pear = Concept(s_pear, 1.0, 10.0, w_pear)
        
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        f_apple = Concept(s_apple, 1.0, 5.0, w_apple)
        
        self.assertAlmostEqual(f_pear.subset_of(f_apple), 0.35158458440000007)
        self.assertAlmostEqual(f_apple.subset_of(f_pear), 0.3125195015901235)

    def test_subset_of_orange_lemon(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_orange = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
        s_orange = Core([c_orange], domains)
        w_orange = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_orange = Concept(s_orange, 1.0, 15.0, w_orange)

        c_lemon = Cuboid([0.7, 0.45, 0.0], [0.8, 0.55, 0.1], domains)
        s_lemon = Core([c_lemon], domains)
        w_lemon = Weights({"color":0.5, "shape":0.5, "taste":2.0}, w_dim)
        f_lemon = Concept(s_lemon, 1.0, 20.0, w_lemon)
        
        self.assertAlmostEqual(f_orange.subset_of(f_lemon), 0.00030722252128279883)


    # implies()
    def test_implies_granny_smith_apple(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
        s_granny_smith = Core([c_granny_smith], domains)
        w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_granny_smith = Concept(s_granny_smith, 1.0, 25.0, w_granny_smith)
        
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        f_apple = Concept(s_apple, 1.0, 5.0, w_apple)
        
        self.assertAlmostEqual(f_granny_smith.implies(f_apple), 1.00)
        self.assertAlmostEqual(f_apple.implies(f_granny_smith), 0.46172839506172847)
    
    def test_implies_lemon_nonSweet(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_lemon = Cuboid([0.7, 0.45, 0.0], [0.8, 0.55, 0.1], domains)
        s_lemon = Core([c_lemon], domains)
        w_lemon = Weights({"color":0.5, "shape":0.5, "taste":2.0}, w_dim)
        f_lemon = Concept(s_lemon, 1.0, 20.0, w_lemon)
        
        c_non_sweet = Cuboid([float("-inf"), float("-inf"), 0.0], [float("inf"), float("inf"), 0.2], {"taste":[2]})
        s_non_sweet = Core([c_non_sweet], {"taste":[2]})
        w_non_sweet = Weights({"taste":1.0}, {"taste":{2:1.0}})
        f_non_sweet = Concept(s_non_sweet, 1.0, 7.0, w_non_sweet)
        
        self.assertAlmostEqual(f_lemon.implies(f_non_sweet), 1.00)
    
    def test_implies_apple_red(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        ConceptualSpace(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        f_apple = Concept(s_apple, 1.0, 5.0, w_apple)
        
        c_red = Cuboid([0.9, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
        s_red = Core([c_red], {"color":[0]})
        w_red = Weights({"color":1.0}, {"color":{0:1.0}})
        f_red = Concept(s_red, 1.0, 20.0, w_red)
        
        self.assertAlmostEqual(f_apple.implies(f_red), 0.6111111111111109)

unittest.main()