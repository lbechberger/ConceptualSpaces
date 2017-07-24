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
import cs.cs as cs

class TestConcept(unittest.TestCase):

    # constructor()
    def test_constructor_fine(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
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
        cs.init(4, {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        with self.assertRaises(Exception):        
            Concept(42, 1.0, 2.0, w)        
        
    def test_constructor_wrong_mu(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 0.0, 2.0, w)        
    
    def test_constructor_wrong_c(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 1.0, -1.0, w)        
            
    def test_constructor_wrong_weigths(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        with self.assertRaises(Exception):        
            Concept(s, 1.0, 1.0, 42)        

    def test_constructor_same_relevant_dimensions(self):
        cs.init(4, {0:[0], 1:[1,2,3]})
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
    

    # membership_of()
    def test_membership_inside(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership_of(p), 1.0)
   
    def test_membership_inside_other_c(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 10.0, w)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership_of(p), 1.0)

    def test_membership_inside_other_mu(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 0.5, 2.0, w)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership_of(p), 0.5)
     
    def test_membership_outside_one_cuboid(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6],{0:[0,1], 1:[2,3]})],{0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 4, 4]
        self.assertAlmostEqual(f.membership_of(p), 0.15173524)

    def test_membership_outside_two_cuboids(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 4, 4]
        self.assertAlmostEqual(f.membership_of(p), 0.15173524)
    
    def test_membership_inside_infinity(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([float("-inf"),float("-inf"),3,4],[float("inf"),float("inf"),5,6], {1:[2,3]})], {1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [1.5, 4, 4, 4]
        self.assertEqual(f.membership_of(p), 1.0)
    
    def test_membership_outside_one_cuboid_infinity(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,float("-inf"),float("-inf")],[3,4,float("inf"),float("inf")], {0:[0,1]})], {0:[0,1]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 10, 4]
        self.assertAlmostEqual(f.membership_of(p), 0.15173524)
        
    def test_membership_outside_two_cuboids_infinity(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([float("-inf"), float("-inf"), 5, 6],[float("inf"), float("inf"), 10, 10], {1:[2,3]}), Cuboid([float("-inf"),float("-inf"),3,4],[float("inf"),float("inf"),5,6], {1:[2,3]})], {1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 10, 4]
        self.assertAlmostEqual(f.membership_of(p), 0.18515757)
    
    # __eq__(), __ne__()
    def test_eq_ne_no_concept(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        self.assertFalse(f == 42)
        self.assertTrue(f != 42)
    
    def test_eq_ne_identity(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        self.assertTrue(f == f)
        self.assertFalse(f != f)

    def test_eq_ne_shallow_copy(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        f2 = Concept(s, 1.0, 2.0, w)
        self.assertTrue(f == f2)
        self.assertFalse(f != f2)
    
    def test_eq_ne_other_params(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
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
     
    # unify_with()
    def test_unify_no_repair_no_params(self):
        cs.init(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f = Concept(Core([c1], {0:[0], 1:[1,2]}), 1.0, 2.0, w) 
        f2 = Concept(Core([c2], {0:[0], 1:[1,2]}), 1.0, 2.0, w)
        f_res = Concept(Core([c1,c2], {0:[0], 1:[1,2]}), 1.0, 2.0, w)
        self.assertEqual(f.unify_with(f2), f_res)
        self.assertEqual(f2.unify_with(f), f_res)

    def test_unify_no_repair_params(self):
        cs.init(3, {0:[0], 1:[1,2]})
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
        self.assertEqual(f.unify_with(f2), f_res)
        self.assertEqual(f2.unify_with(f), f_res)
        
    def test_unify_repair_no_params(self):
        cs.init(3, {0:[0], 1:[1,2]})
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
        self.assertEqual(f.unify_with(f2), f_res)
        self.assertEqual(f2.unify_with(f), f_res)
        
    def test_unify_identity(self):
        doms = {0:[0,1], 1:[2,3]}
        cs.init(4, doms)
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], doms), Cuboid([1,2,3,4],[3,4,5,6], doms)], doms)
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        self.assertEqual(f, f.unify_with(f))
    
    # cut_at()
    def test_cut_above(self):
        cs.init(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        s1 = Core([c1, c2], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f1 = Concept(s1, 1.0, 2.0, w)
        self.assertEqual(f1.cut_at(0,8.0), (f1, None))

    def test_cut_below(self):
        cs.init(3, {0:[0], 1:[1,2]})
        c1 = Cuboid([1,2,3],[7,8,9], {0:[0], 1:[1,2]})
        c2 = Cuboid([4,5,6],[7,7,7], {0:[0], 1:[1,2]})
        s1 = Core([c1, c2], {0:[0], 1:[1,2]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1}, 1:{1:3, 2:2.0}}
        w = Weights(dom, dim)
        f1 = Concept(s1, 1.0, 2.0, w)
        self.assertEqual(f1.cut_at(2,0.0), (None, f1))
        
    def test_cut_through_center(self):
        cs.init(3, {0:[0], 1:[1,2]})
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
        
        self.assertEqual(f1.cut_at(0, 5), (low_f, up_f))

    def test_cut_through_one_cuboid(self):
        cs.init(3, {0:[0], 1:[1,2]})
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
        
        self.assertEqual(f1.cut_at(2, 5), (low_f, up_f))
    
    # project_onto()
    def test_project_identical_domains(self):
        cs.init(3, {0:[0,1,2]})
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        s = Core([c1],{0:[0,1,2]})
        w = Weights({0:1}, {0:{0:0.5, 1:0.3, 2:0.2}})
        f = Concept(s, 1.0, 5.0, w)
        self.assertEqual(f.project_onto({0:[0,1,2]}), f)
    
    def test_project_correct(self):
        cs.init(3, {0:[0,1], 1:[2]})
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
        
        self.assertEqual(f.project_onto({0:[0,1]}), f_res1)
        self.assertEqual(f.project_onto({1:[2]}), f_res2)

    # size()
    def test_hypervolume_single_cuboid_lemon(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_lemon = Cuboid([0.7, 0.45, 0.0], [0.8, 0.55, 0.1], domains)
        s_lemon = Core([c_lemon], domains)
        w_lemon = Weights({"color":0.5, "shape":0.5, "taste":2.0}, w_dim)
        f_lemon = Concept(s_lemon, 1.0, 20.0, w_lemon)
        
        self.assertAlmostEqual(f_lemon.size(), 54.0/4000.0)

    def test_hypervolume_single_cuboid_granny_smith(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
        s_granny_smith = Core([c_granny_smith], domains)
        w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_granny_smith = Concept(s_granny_smith, 1.0, 25.0, w_granny_smith)

        self.assertAlmostEqual(f_granny_smith.size(), 0.004212)

    def test_hypervolume_single_cuboid_pear(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        f_pear = Concept(s_pear, 1.0, 10.0, w_pear)

        self.assertAlmostEqual(f_pear.size(), 0.0561600)
 
    def test_hypervolume_single_cuboid_orange(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_orange = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
        s_orange = Core([c_orange], domains)
        w_orange = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_orange = Concept(s_orange, 1.0, 15.0, w_orange)

        self.assertAlmostEqual(f_orange.size(), 0.01270370)
   
    def test_hypervolume_multiple_cuboids_apple(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        f_apple = Concept(s_apple, 1.0, 5.0, w_apple)

        self.assertAlmostEqual(f_apple.size(), 0.3375000)
    
    # intersect_with()
    # coding: {num_cuboids}_{space_dim}_{space_type}_{intersection_type}_{mu}_{weights}_{c}_{alpha}
    def test_intersect_1C_2D_M_crisp_sameMu_sameW_sameC(self):
        doms = {0:[0], 1:[1]}       
        cs.init(2, doms)
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
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

    def test_intersect_1C_2D_E_crisp_sameMu_sameW_sameC(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
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
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

    def test_intersect_1C_2D_E_crisp_diffMu_diffW_diffC(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
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
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)
 
    def test_intersect_1C_2D_E_muOverlap_diffMu_sameW_sameC(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
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
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

    def test_intersect_1C_2D_E_muOverlap_diffMu_sameW_sameC_variant2(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
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
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

    def test_intersect_1C_2D_E_1diffPoints_sameMu_sameW_sameC(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
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
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

    def test_intersect_1C_2D_E_1diffExtrude_sameMu_sameW_sameC(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
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
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

    def test_intersect_1C_2D_E_2diff_diffMu_diffW_diffC(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
        c1 = Cuboid([0.00,0.00],[0.30,0.25], doms)
        c2 = Cuboid([0.40,0.50],[0.55,0.90], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1}, {0:{0:1, 1:1}})
        w2 = Weights({0:1}, {0:{0:2, 1:1}})
        f1 = Concept(s1, 0.9, 5.0, w1)
        f2 = Concept(s2, 1.0, 8.0, w2)

        c_res1 = Cuboid([0.3670043848, 0.3762729758],[0.3670043848, 0.3762729758], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1}, {0:{0:(7/12.0), 1:(5/12.0)}})        
        f_res1 = Concept(s_res1, 0.5429369989, 5.0, w_res1)
        
        c_res2 = Cuboid([0.3669356482, 0.3763095722],[0.3669356482, 0.3763095722], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1}, {0:{0:(7/12.0), 1:(5/12.0)}})        
        f_res2 = Concept(s_res2, 0.54293668, 5.0, w_res2)
        
        
        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_1C_2D_M_2diff_diffMu_diffW_diffC(self):
        doms = {0:[0],1:[1]}       
        cs.init(2, doms)
        c1 = Cuboid([0.00,0.00],[0.30,0.25], doms)
        c2 = Cuboid([0.40,0.50],[0.55,0.90], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})
        w2 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})
        f1 = Concept(s1, 0.9, 5.0, w1)
        f2 = Concept(s2, 1.0, 8.0, w2)

        c_res1 = Cuboid([0.3999999998, 0.3204489823],[0.3999999998, 0.3204489823], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:(7/12.0), 1:(5/12.0)}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.3838108499, 5.0, w_res1)
        
        c_res2 = Cuboid([0.3999999663, 0.3204490335],[0.3999999663, 0.3204490335], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:(7/12.0), 1:(5/12.0)}, {0:{0:1}, 1:{1:1}})        
        f_res2 = Concept(s_res2, 0.383810816, 5.0, w_res2)
              
        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_1C_2D_M_2diff_diffMu_sameW_diffC(self):
        doms = {0:[0],1:[1]}       
        cs.init(2, doms)
        c1 = Cuboid([0.00,0.00],[0.30,0.25], doms)
        c2 = Cuboid([0.40,0.50],[0.55,0.90], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})
        w2 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})
        f1 = Concept(s1, 0.9, 5.0, w1)
        f2 = Concept(s2, 1.0, 8.0, w2)

        c_res1 = Cuboid([0.3073830488, 0.3147660869],[0.4, 0.5], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.3723525481, 5.0, w_res1)
        
        c_res2 = Cuboid([0.3073830488, 0.3147660869],[0.4, 0.5], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res2 = Concept(s_res2, 0.3723525481, 5.0, w_res2)
        
        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_2C_2D_M_2diff_sameMu_sameW_sameC_sameAlpha(self):
        doms = {0:[0],1:[1]}       
        cs.init(2, doms)
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

        c_res11 = Cuboid([0.20, 0.4999975028],[0.45, 0.8499950019], doms)
        c_res12 = Cuboid([0.45, 0.1500000037],[0.70, 0.4999975028], doms)
        s_res1 = Core([c_res11, c_res12], doms)
        w_res1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.1785041281, 10.0, w_res1)
        
        c_res21 = Cuboid([0.20, 0.4999975028],[0.45, 0.8499950019], doms)
        c_res22 = Cuboid([0.45, 0.1500000037],[0.70, 0.4999975028], doms)
        s_res2 = Core([c_res21, c_res22], doms)
        w_res2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res2 = Concept(s_res2, 0.1785041281, 10.0, w_res2)
        
        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_2C_2D_M_2diff_sameMu_sameW_sameC_diffAlpha(self):
        doms = {0:[0],1:[1]}       
        cs.init(2, doms)
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

        c_res1 = Cuboid([0.60, 0.1500000037],[0.70, 0.2499950019], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.1785041281, 10.0, w_res1)
        
        c_res2 = Cuboid([0.60, 0.1500000037],[0.70, 0.2499950018], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res2 = Concept(s_res2, 0.1785041281, 10.0, w_res2)
        
        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_1C_3D_C_2diffExtrM_sameMu_depW_sameC(self):
        doms = {0:[0,1],1:[2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.40], doms)
        c2 = Cuboid([0.50,0.90,0.30],[1.00,1.00,0.70], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1, 1:2}, {0:{0:3, 1:1}, 1:{2:1}})
        w2 = Weights({0:1, 1:2}, {0:{0:1, 1:1}, 1:{2:1}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)

        c_res1 = Cuboid([0.327200066, 0.8376643818, 0.3],[0.327200066, 0.8376643818, 0.4], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:(2.0/3), 1:(4.0/3)}, {0:{0:0.625, 1:0.375}, 1:{2:1}})
        f_res1 = Concept(s_res1, 0.8409747859, 2.0, w_res1)
        
        c_res2 = Cuboid([0.3272006707, 0.8376627822, 0.3],[0.3272006707, 0.8376627822, 0.4], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1, 1:2}, {0:{0:0.625, 1:0.375}, 1:{2:1}})
        f_res2 = Concept(s_res2, 0.8409747627, 2.0, w_res2)
        
        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_1C_3D_C_2diffExtrE_sameMu_depW_sameC(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.40], doms)
        c2 = Cuboid([0.50,0.90,0.30],[1.00,1.00,0.70], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:(2/3.0), 1:(4/3.0)}, {0:{0:1}, 1:{1:0.125, 2:0.875}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)

        c_res1 = Cuboid([0.3234313871, 0.7, 0.3],[0.4648529177, 0.9, 0.4], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:(5/6.0), 1:(7/6.0)}, {0:{0:1}, 1:{1:0.3125, 2:0.6875}})
        f_res1 = Concept(s_res1, 0.702480783, 2.0, w_res1)
        
        c_res2 = Cuboid([0.3234315726, 0.7, 0.3],[0.464852639, 0.9, 0.4], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:(5/6.0), 1:(7/6.0)}, {0:{0:1}, 1:{1:0.3125, 2:0.6875}})
        f_res2 = Concept(s_res2, 0.7024810437, 2.0, w_res2)
        
        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_1C_3D_C_3diff_sameMu_sameW_sameC(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.30], doms)
        c2 = Cuboid([0.50,0.90,0.40],[1.00,1.00,0.70], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        w2 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)

        c_res1 = Cuboid([0.3279520727, 0.7, 0.3],[0.3720479257, 0.9, 0.4], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        f_res1 = Concept(s_res1, 0.5968175744, 2.0, w_res1)
        
        c_res2 = Cuboid([0.3279520727, 0.7, 0.3],[0.3720479257, 0.9, 0.4], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        f_res2 = Concept(s_res2, 0.5968175744, 2.0, w_res2)

        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_1C_3D_C_3diff_sameMu_diffW_sameC(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.30], doms)
        c2 = Cuboid([0.50,0.90,0.40],[1.00,1.00,0.70], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.6, 2:0.4}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)

        c_res1 = Cuboid([0.293542487, 0.9, 0.3999999996],[0.293542487, 0.9, 0.3999999996], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1.25, 1:0.75}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res1 = Concept(s_res1, 0.6617185092, 2.0, w_res1)
        
        c_res2 = Cuboid([0.293542507, 0.8999997212, 0.4],[0.293542507, 0.8999997212, 0.4], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:1.25, 1:0.75}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res2 = Concept(s_res2, 0.6617182499, 2.0, w_res2)

        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_2C_3D_C_3diffMuOverlap_diffMu_diffW_diffC_diffAlpha(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
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

        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_1C_3D_C_3diffExtr_sameMu_diffW_diffC_2D3Dcuboids(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.20,0.70,0.30], doms)
        c2 = Cuboid([-float("inf"),0.90,0.35],[float("inf"),1.00,0.70], {1:[1,2]})
        s1 = Core([c1], doms)
        s2 = Core([c2],  {1:[1,2]})
        w1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        w2 = Weights({1:1}, {1:{1:0.6, 2:0.4}})
        f1 = Concept(s1, 1.0, 2.0, w1)
        f2 = Concept(s2, 1.0, 5.0, w2)

        c_res1 = Cuboid([0.0, 0.8798304865, 0.3332340033],[0.2, 0.8798304865, 0.3332340033], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:(4.0/3.0), 1:(2.0/3.0)}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res1 = Concept(s_res1, 0.9099102151, 2.0, w_res1)
        
        c_res2 = Cuboid([0.0, 0.8798294352, 0.3332359396],[0.2, 0.8798294352, 0.3332359396], doms)
        s_res2 = Core([c_res2], doms)
        w_res2 = Weights({0:(4.0/3.0), 1:(2.0/3.0)}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res2 = Concept(s_res2, 0.9099102764, 2.0, w_res2)

        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_1C_3D_C_3diffExtr_sameMu_diffW_diffC_2D1Dcuboids(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
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

        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)

    def test_intersect_apple_pear(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
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

        c_res1 = Cuboid([0.5, 0.6187499794, 0.35],[0.7, 0.6187499794, 0.45], domains)
        s_res1 = Core([c_res1], domains)
        w_res1 = Weights({"color":0.50, "shape":1.375, "taste":1.125}, w_dim)
        f_res1 = Concept(s_res1, 0.7910649884, 5.0, w_res1)
        
        c_res2 = Cuboid([0.5, 0.6187499794, 0.35],[0.7, 0.6187499794, 0.45], domains)
        s_res2 = Core([c_res2], domains)
        w_res2 = Weights({"color":0.50, "shape":1.375, "taste":1.125}, w_dim)
        f_res2 = Concept(s_res2, 0.791065315, 5.0, w_res2)
        
        self.assertEqual(f_apple.intersect_with(f_pear), f_res1)
        self.assertEqual(f_pear.intersect_with(f_apple), f_res2)

    def test_intersect_red_green(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        c_red = Cuboid([0.9, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
        s_red = Core([c_red], {"color":[0]})
        w_red = Weights({"color":1.0}, {"color":{0:1.0}})
        f_red = Concept(s_red, 1.0, 20.0, w_red)
        
        c_green = Cuboid([0.45, float("-inf"), float("-inf")], [0.55, float("inf"), float("inf")], {"color":[0]})
        s_green = Core([c_green], {"color":[0]})
        w_green = Weights({"color":1.0}, {"color":{0:1.0}})
        f_green = Concept(s_green, 1.0, 20.0, w_green)

        c_res1 = Cuboid([0.725, float("-inf"), float("-inf")], [0.725, float("inf"), float("inf")], {"color":[0]})
        s_res1 = Core([c_res1], {"color":[0]})
        w_res1 = Weights({"color":1.0}, {"color":{0:1.0}})
        f_res1 = Concept(s_res1, 0.0301973834, 20.0, w_res1)
        
        c_res2 = Cuboid([0.725, float("-inf"), float("-inf")], [0.725, float("inf"), float("inf")], {"color":[0]})
        s_res2 = Core([c_res2], {"color":[0]})
        w_res2 = Weights({"color":1.0}, {"color":{0:1.0}})
        f_res2 = Concept(s_res2, 0.0301973834, 20.0, w_res2)
         
        self.assertEqual(f_red.intersect_with(f_green), f_res1)
        self.assertEqual(f_green.intersect_with(f_red), f_res2)

    def test_intersect_pathological_cones(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.40,0.40], doms)
        c2 = Cuboid([0.60,0.60,0.60],[1.00,1.00,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 1.0, 1.0, w)

        c_res1 = Cuboid([0.4, 0.4857142872, 0.4857142872],[0.6, 0.5142857143, 0.5142857143], doms)
        s_res1 = Core([c_res1], doms)
        f_res1 = Concept(s_res1, 0.8187307531, 1.0, w)
        
        c_res2 = Cuboid([0.4, 0.4857142872, 0.4857142872],[0.6, 0.5142857143, 0.5142857143], doms)
        s_res2 = Core([c_res2], doms)
        f_res2 = Concept(s_res2, 0.8187307531, 1.0, w)

        self.assertEqual(f1.intersect_with(f2), f_res1)
        self.assertEqual(f2.intersect_with(f1), f_res2)
        

    # subset_of()
    def test_subset_of_granny_smith_apple(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
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
        cs.init(3, domains)
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
        
        self.assertAlmostEqual(f_pear.subset_of(f_apple), 0.4520373228571429)
        self.assertAlmostEqual(f_apple.subset_of(f_pear), 0.19069907388897037)

    def test_subset_of_orange_lemon(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_orange = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
        s_orange = Core([c_orange], domains)
        w_orange = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_orange = Concept(s_orange, 1.0, 15.0, w_orange)

        c_lemon = Cuboid([0.7, 0.45, 0.0], [0.8, 0.55, 0.1], domains)
        s_lemon = Core([c_lemon], domains)
        w_lemon = Weights({"color":0.5, "shape":0.5, "taste":2.0}, w_dim)
        f_lemon = Concept(s_lemon, 1.0, 20.0, w_lemon)
        
        self.assertAlmostEqual(f_orange.subset_of(f_lemon), 0.00024392897777777777)


    # implies()
    def test_implies_granny_smith_apple(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
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
        self.assertAlmostEqual(f_apple.implies(f_granny_smith), 0.11709107083287003)
    
    def test_implies_lemon_nonSweet(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
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
        cs.init(3, domains)
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
        
        self.assertAlmostEqual(f_apple.implies(f_red), 0.3333333333333333)

    # similarity_to()

    # 'naive'
    def test_similarity_naive_symmetric(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.40,0.40], doms)
        c2 = Cuboid([0.60,0.60,0.60],[1.00,1.00,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 1.0, 1.0, w)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "naive"), 0.3011942119122)
        self.assertAlmostEqual(f2.similarity_to(f1, "naive"), f1.similarity_to(f2))

    def test_similarity_naive_asymmetric(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.30,0.10], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "naive"), 0.061968893864880734)
        self.assertAlmostEqual(f2.similarity_to(f1, "naive"), 0.23444817280911917)
    
    def test_similarity_naive_apple_red(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, {"color":{0:1}, "shape":{1:1}, "taste":{2:1}})
        f_apple = Concept(s_apple, 1.0, 10.0, w_apple)
        c_red = Cuboid([0.9, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
        s_red = Core([c_red], {"color":[0]})
        w_red = Weights({"color":1.0}, {"color":{0:1.0}})
        f_red = Concept(s_red, 1.0, 20.0, w_red)
        
        self.assertAlmostEqual(f_apple.similarity_to(f_red, "naive"), 0.018315638888734196)
        self.assertAlmostEqual(f_red.similarity_to(f_apple, "naive"), 0.3678794411714424)

    def test_similarity_naive_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "naive"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "naive"), 1.0)

    # 'Jaccard'
    def test_similarity_Jaccard(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.50,0.60], doms)
        c2 = Cuboid([0.40,0.40,0.60],[1.00,1.00,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1.00, 1:1.00}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "Jaccard"), 0.2969857943280167)
        self.assertAlmostEqual(f2.similarity_to(f1, "Jaccard"), 0.4940939754738745)

    def test_similarity_Jaccard_no_overlap(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.30,0.10], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "Jaccard"), 0.18320509717302705)
        self.assertAlmostEqual(f2.similarity_to(f1, "Jaccard"), 0.307282681624636)
    
    def test_similarity_Jaccard_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "Jaccard"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "Jaccard"), 1.0)

    # 'Jaccard_mod'
    def test_similarity_Jaccard_mod(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.50,0.60], doms)
        c2 = Cuboid([0.40,0.40,0.60],[1.00,1.00,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1.00, 1:1.00}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "Jaccard_mod"), 0.2969857943280167)
        self.assertAlmostEqual(f2.similarity_to(f1, "Jaccard_mod"), 0.4940939754738745)

    def test_similarity_Jaccard_mod_no_overlap(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.30,0.10], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "Jaccard_mod"), 0.2412730914306646)
        self.assertAlmostEqual(f2.similarity_to(f1, "Jaccard_mod"), 0.33510692443592327)
    
    def test_similarity_Jaccard_mod_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "Jaccard_mod"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "Jaccard_mod"), 1.0)
        
    # 'subset'
    def test_similarity_subset(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
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
        
        self.assertAlmostEqual(f_pear.subset_of(f_apple), 0.4520373228571429)
        self.assertAlmostEqual(f_apple.subset_of(f_pear), 0.19069907388897037)

    def test_similarity_subset_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "subset"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "subset"), 1.0)

    # 'min_core'
    def test_similarity_min_core(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.00,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "min_core"), 0.20316732)
        self.assertAlmostEqual(f2.similarity_to(f1, "min_core"), 0.4636068)

    def test_similarity_min_core_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "min_core"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "min_core"), 1.0)

    # 'max_core'
    def test_similarity_max_core(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "max_core"), 0.018315638888734186)
        self.assertAlmostEqual(f2.similarity_to(f1, "max_core"), 0.1353352832366127)

    # 'Hausdorff_core'
    def test_similarity_Hausdorff_core(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "Hausdorff_core"), 0.05767125507091008)
        self.assertAlmostEqual(f2.similarity_to(f1, "Hausdorff_core"), 0.23753581495253084)
    
    def test_similarity_Hausdorff_core_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "Hausdorff_core"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "Hausdorff_core"), 1.0)
        
    # 'min_membership_core'
    def test_similarity_min_membership_core(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "min_membership_core"), 0.05767125507091008)
        self.assertAlmostEqual(f2.similarity_to(f1, "min_membership_core"), 0.23753581495253084)            

    def test_similarity_min_membership_core_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "min_membership_core"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "min_membership_core"), 1.0)

    # 'max_membership_core'
    def test_similarity_max_membership_core(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "max_membership_core"), 0.20316732)
        self.assertAlmostEqual(f2.similarity_to(f1, "max_membership_core"), 0.4636068)            

    def test_similarity_max_membership_core_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "max_membership_core"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "max_membership_core"), 1.0)
    
    # 'min_center'
    def test_similarity_min_center(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.00,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "min_center"), 0.12045065174829248)
        self.assertAlmostEqual(f2.similarity_to(f1, "min_center"), 0.35263265020521906)

    def test_similarity_min_center_identity(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "min_center"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "min_center"), 1.0)

    # 'max_center'
    def test_similarity_max_center(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "max_center"), 0.021236669082903312)
        self.assertAlmostEqual(f2.similarity_to(f1, "max_center"), 0.14737115075690613)
    
    # 'Hausdorff_center'
    def test_similarity_Hausdorff_center(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "Hausdorff_center"), 0.03868624516619911)
        self.assertAlmostEqual(f2.similarity_to(f1, "Hausdorff_center"), 0.1845063669249684)

    def test_similarity_Hausdorff_center_identity(self):
        doms = {0:[0],1:[1,2]}
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "Hausdorff_center"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "Hausdorff_center"), 1.0)
    
    # 'min_membership_center'
    def test_similarity_min_membership_center(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "min_membership_center"), 0.0663008293667595)
        self.assertAlmostEqual(f2.similarity_to(f1, "min_membership_center"), 0.23753581495253084)
        
    def test_similarity_min_membership_center_identity(self):
        doms = {0:[0],1:[1,2]}
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "min_membership_center"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "min_membership_center"), 1.0)

    # 'max_membership_center'
    def test_similarity_max_membership_center(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertAlmostEqual(f1.similarity_to(f2, "max_membership_center"), 0.12045065174829248)
        self.assertAlmostEqual(f2.similarity_to(f1, "max_membership_center"), 0.4636068)
        
    def test_similarity_max_membership_center_identity(self):
        doms = {0:[0],1:[1,2]}
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.10,0.00],[0.40,0.30,0.10], doms)
        c1_2 = Cuboid([0.00,0.00,0.00],[0.20,0.20,0.40], doms)
        c2 = Cuboid([0.80,0.60,0.70],[1.00,1.00,1.00], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2], doms)
        w1 = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w2 = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:0.75, 2:0.25}})
        f1 = Concept(s1, 1.0, 1.0, w1)
        f2 = Concept(s2, 1.0, 2.0, w2)
        
        self.assertEqual(f1.similarity_to(f1, "max_membership_center"), 1.0)
        self.assertEqual(f2.similarity_to(f2, "max_membership_center"), 1.0)
    
    # between()
    # 'naive'
    def test_between_naive(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.40,0.40], doms)
        c2 = Cuboid([0.60,0.60,0.60],[1.00,1.00,1.00], doms)
        c3 = Cuboid([0.30,0.40,0.30],[0.40,0.40,0.50], doms)
        c4 = Cuboid([0.30,0.30,0.30],[0.40,0.40,0.50], doms)
        c5_1 = Cuboid([0.8, 1.1, 0.8],[2.0, 1.6, 1.4], doms)
        c5_2 = Cuboid([1.4, 0.9, 1.3],[2.9, 1.5, 2.0], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        s3 = Core([c3], doms)
        s4 = Core([c4], doms)
        s5 = Core([c5_1,c5_2],doms)
        w = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 1.0, 1.0, w)
        f3 = Concept(s3, 1.0, 1.0, w)
        f4 = Concept(s4, 1.0, 1.0, w)
        f5 = Concept(s5, 1.0, 1.0, w)
        
        self.assertAlmostEqual(f1.between(f1, f1, method="naive"), 1.0)
        self.assertAlmostEqual(f3.between(f1, f2, method="naive"), 1.0)
        self.assertAlmostEqual(f3.between(f2, f1, method="naive"), 1.0)
        self.assertAlmostEqual(f4.between(f1, f2, method="naive"), 0.0)
        self.assertAlmostEqual(f4.between(f2, f1, method="naive"), 0.0)
        self.assertAlmostEqual(f5.between(f1, f2, method="naive"), 0.0)
        self.assertAlmostEqual(f5.between(f2, f1, method="naive"), 0.0)

    # 'naive_soft'
    def test_between_naive_soft(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.40,0.40], doms)
        c2 = Cuboid([0.60,0.60,0.60],[1.00,1.00,1.00], doms)
        c3 = Cuboid([0.30,0.40,0.30],[0.40,0.40,0.50], doms)
        c4 = Cuboid([0.30,0.30,0.30],[0.40,0.40,0.50], doms)
        c5_1 = Cuboid([0.8, 1.1, 0.8],[2.0, 1.6, 1.4], doms)
        c5_2 = Cuboid([1.4, 0.9, 1.3],[2.9, 1.5, 2.0], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        s3 = Core([c3], doms)
        s4 = Core([c4], doms)
        s5 = Core([c5_1,c5_2],doms)
        w = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 1.0, 1.0, w)
        f3 = Concept(s3, 1.0, 1.0, w)
        f4 = Concept(s4, 1.0, 1.0, w)
        f5 = Concept(s5, 1.0, 1.0, w)
        
        self.assertAlmostEqual(f1.between(f1, f1, method="naive_soft"), 1.0)
        self.assertAlmostEqual(f3.between(f1, f2, method="naive_soft"), 1.0)
        self.assertAlmostEqual(f3.between(f2, f1, method="naive_soft"), 1.0)
        self.assertAlmostEqual(f4.between(f1, f2, method="naive_soft"), 0.996350972)
        self.assertAlmostEqual(f4.between(f2, f1, method="naive_soft"), 0.996350972)
        self.assertAlmostEqual(f5.between(f1, f2, method="naive_soft"), 0.343935417)
        self.assertAlmostEqual(f5.between(f2, f1, method="naive_soft"), 0.343935417)

    # 'subset'
    def test_between_subset(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.40,0.40], doms)
        c2 = Cuboid([0.60,0.60,0.60],[1.00,1.00,1.00], doms)
        c3 = Cuboid([0.30,0.40,0.30],[0.40,0.40,0.50], doms)
        c4 = Cuboid([0.30,0.30,0.30],[0.40,0.40,0.50], doms)
        c5_1 = Cuboid([0.8, 1.1, 0.8],[2.0, 1.6, 1.4], doms)
        c5_2 = Cuboid([1.4, 0.9, 1.3],[2.9, 1.5, 2.0], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        s3 = Core([c3], doms)
        s4 = Core([c4], doms)
        s5 = Core([c5_1,c5_2],doms)
        w = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 1.0, 1.0, w)
        f3 = Concept(s3, 1.0, 1.0, w)
        f4 = Concept(s4, 1.0, 1.0, w)
        f5 = Concept(s5, 1.0, 1.0, w)
        
        self.assertAlmostEqual(f1.between(f1, f1, method="subset"), 1.0)
        self.assertAlmostEqual(f3.between(f1, f2, method="subset"), 1.0)
        self.assertAlmostEqual(f3.between(f2, f1, method="subset"), 1.0)
        self.assertAlmostEqual(f4.between(f1, f2, method="subset"), 1.0)
        self.assertAlmostEqual(f4.between(f2, f1, method="subset"), 1.0)
        self.assertAlmostEqual(f5.between(f1, f2, method="subset"), 0.4418742326582702)
        self.assertAlmostEqual(f5.between(f2, f1, method="subset"), 0.4418742326582702)

    # 'core'
    def test_between_core(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.40,0.40], doms)
        c2 = Cuboid([0.60,0.60,0.60],[1.00,1.00,1.00], doms)
        c3 = Cuboid([0.30,0.40,0.30],[0.40,0.40,0.50], doms)
        c4 = Cuboid([0.30,0.30,0.30],[0.40,0.40,0.50], doms)
        c5_1 = Cuboid([0.8, 1.1, 0.8],[2.0, 1.6, 1.4], doms)
        c5_2 = Cuboid([1.4, 0.9, 1.3],[2.9, 1.5, 2.0], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        s3 = Core([c3], doms)
        s4 = Core([c4], doms)
        s5 = Core([c5_1,c5_2],doms)
        w = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 1.0, 1.0, w)
        f3 = Concept(s3, 1.0, 1.0, w)
        f4 = Concept(s4, 1.0, 1.0, w)
        f5 = Concept(s5, 1.0, 1.0, w)
        
        self.assertAlmostEqual(f1.between(f1, f1, method="core"), 1.0)
        self.assertAlmostEqual(f3.between(f1, f2, method="core"), 1.0)
        self.assertAlmostEqual(f3.between(f2, f1, method="core"), 1.0)
        self.assertAlmostEqual(f4.between(f1, f2, method="core"), 1.0)
        self.assertAlmostEqual(f4.between(f2, f1, method="core"), 1.0)
        self.assertAlmostEqual(f5.between(f1, f2, method="core"), 0.0)
        self.assertAlmostEqual(f5.between(f2, f1, method="core"), 0.0)

    def test_between_core_two_cuboids_3d(self):
        doms = {0:[0,1,2]}       
        cs.init(3, doms)
        c1_1 = Cuboid([0.00,0.70,0.90],[0.40,1.00,1.00], doms)
        c1_2 = Cuboid([0.00,0.50,0.70],[0.20,1.00,1.00], doms)
        c2_1 = Cuboid([0.80,0.00,0.00],[1.00,0.30,0.40], doms)
        c2_2 = Cuboid([0.80,0.00,0.00],[1.00,0.50,0.20], doms)
        c3 = Cuboid([0.40,0.40,0.40],[0.70,0.70,0.50], doms)
        c4 = Cuboid([0.40,0.40,0.30],[0.70,0.70,0.50], doms)
        c5_1 = Cuboid([0.40,0.40,0.40],[0.70,0.70,0.50], doms)
        c5_2 = Cuboid([0.40,0.40,0.40],[0.50,0.50,1.00], doms)
        c6 = Cuboid([0.40, 0.40, 0.40], [0.60, 0.70, 1.20], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2_1, c2_2], doms)
        s3 = Core([c3], doms)
        s4 = Core([c4], doms)
        s5 = Core([c5_1,c5_2],doms)
        s6 = Core([c6], doms)
        w = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 1.0, 1.0, w)
        f3 = Concept(s3, 1.0, 1.0, w)
        f4 = Concept(s4, 1.0, 1.0, w)
        f5 = Concept(s5, 1.0, 1.0, w)
        f6 = Concept(s6, 0.5, 5.0, w)
        
        self.assertAlmostEqual(f1.between(f1, f1, method="core"), 1.0)
        self.assertAlmostEqual(f1.between(f1, f2, method="core"), 1.0)
        self.assertAlmostEqual(f3.between(f1, f2, method="core"), 1.0)
        self.assertAlmostEqual(f3.between(f2, f1, method="core"), 1.0)
        self.assertAlmostEqual(f4.between(f1, f2, method="core"), 0.0)
        self.assertAlmostEqual(f4.between(f2, f1, method="core"), 0.0)
        self.assertAlmostEqual(f5.between(f1, f2, method="core"), 0.0)
        self.assertAlmostEqual(f5.between(f2, f1, method="core"), 0.0)
        self.assertAlmostEqual(f6.between(f1, f2, method="core"), 0.0)
        self.assertAlmostEqual(f6.between(f2, f1, method="core"), 0.0)

    def test_between_core_two_cuboids_2d(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
        c1_1 = Cuboid([0.10,0.80],[0.40,0.90], doms)
        c1_2 = Cuboid([0.10,0.60],[0.20,1.00], doms)
        c2_1 = Cuboid([0.70,0.10],[0.90,0.40], doms)
        c2_2 = Cuboid([0.80,0.30],[1.00,0.70], doms)
        c3_1 = Cuboid([0.40,0.40],[0.70,0.50], doms)
        c3_2 = Cuboid([0.60,0.30],[0.70,0.80], doms)
        c4_1 = Cuboid([0.40,0.40],[0.70,0.50], doms)
        c4_2 = Cuboid([0.60,0.30],[0.70,0.90], doms)
        c5 = Cuboid([0.90,0.20],[1.00,0.30], doms)
        s1 = Core([c1_1, c1_2], doms)
        s2 = Core([c2_1, c2_2], doms)
        s3 = Core([c3_1, c3_2], doms)
        s4 = Core([c4_1, c4_2], doms)
        s5 = Core([c5], doms)
        w = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        f1 = Concept(s1, 1.0, 1.0, w)
        f2 = Concept(s2, 1.0, 1.0, w)
        f3 = Concept(s3, 0.9, 15.0, w)
        f4 = Concept(s4, 0.9, 15.0, w)
        f5 = Concept(s5, 0.9, 15.0, w)
        
        self.assertAlmostEqual(f1.between(f1, f1, method="core"), 1.0)
        self.assertAlmostEqual(f1.between(f1, f2, method="core"), 1.0)
        self.assertAlmostEqual(f3.between(f1, f2, method="core"), 1.0)
        self.assertAlmostEqual(f3.between(f2, f1, method="core"), 1.0)
        self.assertAlmostEqual(f4.between(f1, f2, method="core"), 0.0)
        self.assertAlmostEqual(f4.between(f2, f1, method="core"), 0.0)
        self.assertAlmostEqual(f5.between(f1, f2, method="core"), 0.0)
        self.assertAlmostEqual(f5.between(f2, f1, method="core"), 0.0)

unittest.main()