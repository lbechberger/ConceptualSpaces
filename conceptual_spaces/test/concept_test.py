# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:56:30 2017

@author: lbechberger
"""

import unittest
import sys
import random
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
        self.assertAlmostEqual(f.membership_of(p), 0.15173524, places=4)

    def test_membership_outside_two_cuboids(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], {0:[0,1], 1:[2,3]}), Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 4, 4]
        self.assertAlmostEqual(f.membership_of(p), 0.15173524, places=4)
    
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
        self.assertAlmostEqual(f.membership_of(p), 0.15173524, places=4)
        
    def test_membership_outside_two_cuboids_infinity(self):
        cs.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([float("-inf"), float("-inf"), 5, 6],[float("inf"), float("inf"), 10, 10], {1:[2,3]}), Cuboid([float("-inf"),float("-inf"),3,4],[float("inf"),float("inf"),5,6], {1:[2,3]})], {1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 1.0, 2.0, w)  
        p = [4, 4, 10, 4]
        self.assertAlmostEqual(f.membership_of(p), 0.18515757, places=4)
    
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
    
    # union_with()
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
        self.assertEqual(f.union_with(f2), f_res)
        self.assertEqual(f2.union_with(f), f_res)

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
        self.assertEqual(f.union_with(f2), f_res)
        self.assertEqual(f2.union_with(f), f_res)
        
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
        self.assertEqual(f.union_with(f2), f_res)
        self.assertEqual(f2.union_with(f), f_res)
        
    def test_unify_identity(self):
        doms = {0:[0,1], 1:[2,3]}
        cs.init(4, doms)
        s = Core([Cuboid([3, 4, 5, 6],[10, 10, 10, 10], doms), Cuboid([1,2,3,4],[3,4,5,6], doms)], doms)
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        f = Concept(s, 1.0, 2.0, w)  
        self.assertEqual(f, f.union_with(f))
    
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
        
        self.assertAlmostEqual(f_lemon.size(), 54.0/4000.0, places=4)

    def test_hypervolume_single_cuboid_granny_smith(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
        s_granny_smith = Core([c_granny_smith], domains)
        w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_granny_smith = Concept(s_granny_smith, 1.0, 25.0, w_granny_smith)

        self.assertAlmostEqual(f_granny_smith.size(), 0.004212, places=4)

    def test_hypervolume_single_cuboid_pear(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        f_pear = Concept(s_pear, 1.0, 10.0, w_pear)

        self.assertAlmostEqual(f_pear.size(), 0.0561600, places=4)

    def test_hypervolume_single_cuboid_orange(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        c_orange = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
        s_orange = Core([c_orange], domains)
        w_orange = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        f_orange = Concept(s_orange, 1.0, 15.0, w_orange)

        self.assertAlmostEqual(f_orange.size(), 0.01270370, places=4)
  
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

        self.assertAlmostEqual(f_apple.size(), 0.3375000, places=4)
    
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

    def test_intersect_1C_2D_E_muOverlap_diffMu_diffW_diffC(self):
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
        
        c_res = Cuboid([0.45,-5],[0.4794017445, -3.9759935761999996], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:1}, {0:{0:0.45, 1:0.55}})
        f_res = Concept(s_res, 0.8, 3.0, w_res)
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)


    def test_intersect_1C_1D_E_muOverlap_diffMu_sameW_sameC(self):
        doms = {0:[0]}       
        cs.init(1, doms)
        c1 = Cuboid([4],[7], doms)
        c2 = Cuboid([3],[8], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1}, {0:{0:1}})
        f1 = Concept(s1, 1.0, 0.5, w)
        f2 = Concept(s2, 0.5, 0.5, w)

        self.assertEqual(f1.intersect_with(f2), f2)
        self.assertEqual(f2.intersect_with(f1), f2)

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

        c_res = Cuboid([0.5,0.50],[0.6715762173,0.6715762173], doms)
        s_res = Core([c_res], doms)
        f_res = Concept(s_res, 0.5, 2.0, w)
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

    def test_intersect_1C_2D_E_muOverlap_diffMu_sameW_sameC_variant3(self):
        doms = {0:[0,1]}       
        cs.init(2, doms)
        c1 = Cuboid([0.00,0.00],[0.25,0.60], doms)
        c2 = Cuboid([0.50,0.50],[0.75,1.00], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1}, {0:{0:1, 1:1}})
        f1 = Concept(s1, 1.0, 2.0, w)
        f2 = Concept(s2, 0.5, 2.0, w)
        c_res = Cuboid([0.5,0.50],[0.7401290722,1.0], doms)
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

        c_res = Cuboid([0.3670043805, 0.3762729781],[0.3670043805, 0.3762729781], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:1}, {0:{0:(7/12.0), 1:(5/12.0)}})        
        f_res = Concept(s_res, 0.5429369989, 5.0, w_res)

        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

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

        c_res = Cuboid([0.4, 0.3204489824],[0.4, 0.3204489824], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:(7/12.0), 1:(5/12.0)}, {0:{0:1}, 1:{1:1}})        
        f_res = Concept(s_res, 0.3838108494, 5.0, w_res)

        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

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

        c_res1 = Cuboid([0.3073830472, 0.3147660943],[0.4, 0.5], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:2, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res1 = Concept(s_res1, 0.3723525481, 5.0, w_res1)
        
        c_res2 = Cuboid([0.3073830472, 0.3147660943],[0.4, 0.5], doms)
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

        c_res1 = Cuboid([0.20, 0.49999874999999994],[0.44999999999999996, 0.849995], doms)
        c_res2 = Cuboid([0.44999999999999996, 0.15],[0.7, 0.49999874999999994], doms)
        s_res = Core([c_res1, c_res2], doms)
        w_res = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res = Concept(s_res, 0.1785041281, 10.0, w_res)
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

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

        
        c_res = Cuboid([0.60, 0.15],[0.70, 0.25], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:1, 1:1}, {0:{0:1}, 1:{1:1}})        
        f_res = Concept(s_res, 0.1785041281, 10.0, w_res)
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

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

        c_res = Cuboid([0.3272001322, 0.8376641984, 0.3],[0.3272001322, 0.8376641984, 0.4], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:(2.0/3), 1:(4.0/3)}, {0:{0:0.625, 1:0.375}, 1:{2:1}})
        f_res = Concept(s_res, 0.8409747859, 2.0, w_res)
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

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

        c_res = Cuboid([0.3234315689, 0.7, 0.3],[0.4648526467, 0.9, 0.4], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:(5/6.0), 1:(7/6.0)}, {0:{0:1}, 1:{1:0.3125, 2:0.6875}})
        f_res = Concept(s_res, 0.7024810437, 2.0, w_res)
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

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

        c_res1 = Cuboid([0.3279520724, 0.7, 0.3],[0.3720479276, 0.9, 0.4], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1.5, 1:0.5}, {0:{0:1}, 1:{1:0.25, 2:0.75}})
        f_res1 = Concept(s_res1, 0.5968175744, 2.0, w_res1)
        
        c_res2 = Cuboid([0.3279520724, 0.7, 0.3],[0.3720479276, 0.9, 0.4], doms)
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

        c_res = Cuboid([0.293542487, 0.9, 0.3999999996],[0.293542487, 0.9, 0.3999999996], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:1.25, 1:0.75}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res = Concept(s_res, 0.6617185092, 2.0, w_res)
        
        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

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

        c_res1 = Cuboid([0.90, 0.30, 0.30],[0.9090862907, 0.3864920434, 0.3351286822], doms)
        s_res1 = Core([c_res1], doms)
        w_res1 = Weights({0:1.25, 1:0.75}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res1 = Concept(s_res1, 0.5, 5.0, w_res1)
        
        c_res2 = Cuboid([0.90, 0.30, 0.30],[0.9090862907, 0.3864920434, 0.3351286822], doms)
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

        c_res = Cuboid([0.0, 0.8798294309, 0.3332359474],[0.2, 0.8798294309, 0.3332359474], doms)
        s_res = Core([c_res], doms)
        w_res = Weights({0:(4.0/3.0), 1:(2.0/3.0)}, {0:{0:1}, 1:{1:0.425, 2:0.575}})
        f_res = Concept(s_res, 0.9099102765, 2.0, w_res)

        self.assertEqual(f1.intersect_with(f2), f_res)
        self.assertEqual(f2.intersect_with(f1), f_res)

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

        c_res = Cuboid([0.5, 0.6187499794, 0.35],[0.7, 0.6187499794, 0.45], domains)
        s_res = Core([c_res], domains)
        w_res = Weights({"color":0.50, "shape":1.375, "taste":1.125}, w_dim)
        f_res = Concept(s_res, 0.791065315, 5.0, w_res)
        
        self.assertEqual(f_apple.intersect_with(f_pear), f_res)
        self.assertEqual(f_pear.intersect_with(f_apple), f_res)

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

        c_res1 = Cuboid([0.4, 0.4857142857, 0.4857142857],[0.6, 0.5142857143, 0.5142857143], doms)
        s_res1 = Core([c_res1], doms)
        f_res1 = Concept(s_res1, 0.8187307531, 1.0, w)
        
        c_res2 = Cuboid([0.4, 0.4857142857, 0.4857142857],[0.6, 0.5142857143, 0.5142857143], doms)
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
        
        self.assertAlmostEqual(f_granny_smith.subset_of(f_apple), 1.00, places=4)

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
        
        self.assertAlmostEqual(f_pear.subset_of(f_apple), 0.4520373228571429, places=4)
        self.assertAlmostEqual(f_apple.subset_of(f_pear), 0.19069907388897037, places=4)

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
        
        self.assertAlmostEqual(f_orange.subset_of(f_lemon), 0.00024392897777777777, places=4)

    def test_subset_of_red_non_sweet(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        
        # define nonSweet property
        c_non_sweet = Cuboid([float("-inf"), float("-inf"), 0.0], [float("inf"), float("inf"), 0.2], {"taste":[2]})
        s_non_sweet = Core([c_non_sweet], {"taste":[2]})
        w_non_sweet = Weights({"taste":1.0}, {"taste":{2:1.0}})
        f_non_sweet = Concept(s_non_sweet, 1.0, 14.0, w_non_sweet)
        
        # define red property
        c_red = Cuboid([0.9, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
        s_red = Core([c_red], {"color":[0]})
        w_red = Weights({"color":1.0}, {"color":{0:1.0}})
        f_red = Concept(s_red, 1.0, 40.0, w_red)
        
        self.assertEqual(f_red.subset_of(f_non_sweet), 0)



    # crisp_subset_of()
    def test_crisp_subset_of_granny_smith_apple(self):
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
        
        self.assertTrue(f_granny_smith.crisp_subset_of(f_apple))
        self.assertFalse(f_apple.crisp_subset_of(f_granny_smith))

    def test_crisp_subset_of_pear_apple(self):
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
        
        self.assertFalse(f_pear.crisp_subset_of(f_apple))
        self.assertFalse(f_apple.crisp_subset_of(f_pear))

    def test_crisp_subset_of_orange_lemon(self):
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
        
        self.assertFalse(f_orange.crisp_subset_of(f_lemon))
        self.assertFalse(f_lemon.crisp_subset_of(f_orange))
    
    def test_crisp_subset_of_artificial_examples(self):
        domains = {0:[0,1], 1:[2]}
        cs.init(3, domains)
        
        c1 = Cuboid([0,0,0], [0.5,0.5,0.5], domains)
        s1 = Core([c1], domains)
        w1 = Weights({0:1, 1:1}, {0:{0:0.5, 1:0.5}, 1:{2:1}})
        f1 = Concept(s1, 1.0, 5.0, w1)
        
        f2 = Concept(s1, 0.9, 5.0, w1)
        f3 = Concept(s1, 1.0, 4.0, w1)
        
        w4 = Weights({0:0.5, 1:1.5}, {0:{0:0.5, 1:0.5}, 1:{2:1}})
        f4 = Concept(s1, 1.0, 5.0, w4)
        w5 = Weights({0:1, 1:1}, {0:{0:0.75, 1:0.25}, 1:{2:1}})
        f5 = Concept(s1, 1.0, 5.0, w5)
        f6 = Concept(s1, 1.0, 10.0, w5)
        
        c2 = Cuboid([0.5,0.5,0.5], [0.5, 0.5, 0.55], domains)
        s2 = Core([c2], domains)
        f7 = Concept(s2, 0.5, 5.0, w1)
        
        self.assertTrue(f2.crisp_subset_of(f1))
        self.assertFalse(f1.crisp_subset_of(f2))
        self.assertFalse(f3.crisp_subset_of(f1))
        self.assertTrue(f1.crisp_subset_of(f3))
        self.assertFalse(f4.crisp_subset_of(f1))
        self.assertFalse(f1.crisp_subset_of(f4))
        self.assertFalse(f5.crisp_subset_of(f1))
        self.assertFalse(f1.crisp_subset_of(f5))
        self.assertTrue(f6.crisp_subset_of(f1))
        self.assertFalse(f1.crisp_subset_of(f6))
        self.assertTrue(f7.crisp_subset_of(f1))
        self.assertFalse(f1.crisp_subset_of(f7))
        


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
        
        self.assertAlmostEqual(f_granny_smith.implies(f_apple), 1.00, places=4)
        self.assertAlmostEqual(f_apple.implies(f_granny_smith), 0.11709107083287003, places=4)
    
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
        
        self.assertAlmostEqual(f_lemon.implies(f_non_sweet), 1.00, places=4)
    
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
        
        self.assertAlmostEqual(f_apple.implies(f_red), 0.3333333333333333, places=4)

    # similarity_to()

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
        
        self.assertAlmostEqual(f1.similarity_to(f2, "Jaccard"), 0.488486914178, places=4)
        self.assertAlmostEqual(f2.similarity_to(f1, "Jaccard"), 0.488486914178, places=4)

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
        
        self.assertAlmostEqual(f1.similarity_to(f2, "Jaccard"), 0.305151362666, places=4)
        self.assertAlmostEqual(f2.similarity_to(f1, "Jaccard"), 0.305151449644, places=4)
    
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

    def test_similarity_Jaccard_properties(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        pear = Concept(s_pear, 1.0, 12.0, w_pear)
        
        c_non_sweet = Cuboid([float("-inf"), float("-inf"), 0.0], [float("inf"), float("inf"), 0.2], {"taste":[2]})
        s_non_sweet = Core([c_non_sweet], {"taste":[2]})
        w_non_sweet = Weights({"taste":1.0}, {"taste":{2:1.0}})
        non_sweet = Concept(s_non_sweet, 1.0, 7.0, w_non_sweet)

        c_red = Cuboid([0.9, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
        s_red = Core([c_red], {"color":[0]})
        w_red = Weights({"color":1.0}, {"color":{0:1.0}})
        red = Concept(s_red, 1.0, 20.0, w_red)

        c_green = Cuboid([0.45, float("-inf"), float("-inf")], [0.55, float("inf"), float("inf")], {"color":[0]})
        s_green = Core([c_green], {"color":[0]})
        w_green = Weights({"color":1.0}, {"color":{0:1.0}})
        green = Concept(s_green, 1.0, 20.0, w_green)
        
        self.assertAlmostEqual(pear.similarity_to(green, "Jaccard"), 0.52, places=4)
        self.assertAlmostEqual(green.similarity_to(pear, "Jaccard"), 0.52, places=4)
        self.assertAlmostEqual(pear.similarity_to(red, "Jaccard"), 0.055782539925, places=4)
        self.assertAlmostEqual(red.similarity_to(pear, "Jaccard"), 0.055782539925, places=4)
        self.assertAlmostEqual(pear.similarity_to(non_sweet, "Jaccard"), 0.200086115262, places=4)
        self.assertAlmostEqual(non_sweet.similarity_to(pear, "Jaccard"), 0.200086115262, places=4)
        self.assertAlmostEqual(non_sweet.similarity_to(red, "Jaccard"), 0.0, places=4)
        self.assertAlmostEqual(red.similarity_to(non_sweet, "Jaccard"), 0.0, places=4)

    def test_similarity_Jaccard_combined_space_properties(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.40,0.40], doms)
        c2 = Cuboid([0.60,0.60,0.60],[1.00,1.00,1.00], doms)
        c3 = Cuboid([float("-inf"),0.40,0.30],[float("inf"),0.40,0.50], {1:[1,2]})
        c4 = Cuboid([0.30,float("-inf"),float("-inf")],[0.40,float("inf"),float("inf")], {0:[0]})
        c5 = Cuboid([1.20,float("-inf"),float("-inf")],[1.40,float("inf"),float("inf")], {0:[0]})
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        s3 = Core([c3], {1:[1,2]})
        s4 = Core([c4], {0:[0]})
        s5 = Core([c5], {0:[0]})
        w_a = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w_b = Weights({1:1}, {1:{1:0.75, 2:0.25}})
        w_c = Weights({0:1}, {0:{0:1}})
        f1 = Concept(s1, 1.0, 1.0, w_a)
        f2 = Concept(s2, 1.0, 1.0, w_a)
        f3 = Concept(s3, 1.0, 1.0, w_b)
        f4 = Concept(s4, 1.0, 1.0, w_c)
        f5 = Concept(s5, 1.0, 1.0, w_c)
        
        self.assertAlmostEqual(f1.similarity_to(f1, method="Jaccard"), 1.0, places=4)
        self.assertAlmostEqual(f1.similarity_to(f2, method="Jaccard"), 0.410357218383, places=4)
        self.assertAlmostEqual(f2.similarity_to(f1, method="Jaccard"), 0.410357218383, places=4)
        self.assertAlmostEqual(f1.similarity_to(f3, method="Jaccard"), 0.842243453275, places=4)
        self.assertAlmostEqual(f3.similarity_to(f1, method="Jaccard"), 0.842243453275, places=4)
        self.assertAlmostEqual(f3.similarity_to(f2, method="Jaccard"), 0.7056811142505014, places=4)
        self.assertAlmostEqual(f2.similarity_to(f3, method="Jaccard"), 0.705680932556, places=4)
        self.assertAlmostEqual(f4.similarity_to(f1, method="Jaccard"), 0.875, places=4)
        self.assertAlmostEqual(f1.similarity_to(f4, method="Jaccard"), 0.875, places=4)
        self.assertAlmostEqual(f5.similarity_to(f1, method="Jaccard"), 0.394305909412, places=4)
        self.assertAlmostEqual(f1.similarity_to(f5, method="Jaccard"), 0.394305909412, places=4)
        self.assertAlmostEqual(f5.similarity_to(f2, method="Jaccard"), 0.646312441429, places=4)
        self.assertAlmostEqual(f2.similarity_to(f5, method="Jaccard"), 0.646312441429, places=4)

    def test_similarity_Jaccard_concepts(self):

        doms = {0:[0], 1:[1], 2:[2]}
        cs.init(3, doms)
        c1 = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], doms)
        c2 = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], doms)
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        w = Weights({0:1, 1:1, 2:1}, {0:{0:1}, 1:{1:1}, 2:{2:1}})
        f1 = Concept(s1, 1.0, 50.0, w)
        f2 = Concept(s2, 1.0, 30.0, w)

        self.assertAlmostEqual(f2.similarity_to(f1, "Jaccard"), 7.359063748437519e-05, places=4)
        self.assertAlmostEqual(f1.similarity_to(f2, "Jaccard"), 7.366064462823261e-05, places=4)


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
        
        self.assertAlmostEqual(f_pear.subset_of(f_apple), 0.4520373228571429, places=4)
        self.assertAlmostEqual(f_apple.subset_of(f_pear), 0.19069907388897037, places=4)

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

    def test_similarity_subset_properties(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        cs.init(3, domains)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        pear = Concept(s_pear, 1.0, 12.0, w_pear)
        
        c_non_sweet = Cuboid([float("-inf"), float("-inf"), 0.0], [float("inf"), float("inf"), 0.2], {"taste":[2]})
        s_non_sweet = Core([c_non_sweet], {"taste":[2]})
        w_non_sweet = Weights({"taste":1.0}, {"taste":{2:1.0}})
        non_sweet = Concept(s_non_sweet, 1.0, 7.0, w_non_sweet)

        c_red = Cuboid([0.9, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
        s_red = Core([c_red], {"color":[0]})
        w_red = Weights({"color":1.0}, {"color":{0:1.0}})
        red = Concept(s_red, 1.0, 20.0, w_red)

        c_green = Cuboid([0.45, float("-inf"), float("-inf")], [0.55, float("inf"), float("inf")], {"color":[0]})
        s_green = Core([c_green], {"color":[0]})
        w_green = Weights({"color":1.0}, {"color":{0:1.0}})
        green = Concept(s_green, 1.0, 20.0, w_green)
        
        self.assertAlmostEqual(pear.similarity_to(green, "subset"), 0.5, places=4)
        self.assertAlmostEqual(green.similarity_to(pear, "subset"), 0.8125, places=4)
        self.assertAlmostEqual(pear.similarity_to(red, "subset"), 0.07437671989999999, places=4)
        self.assertAlmostEqual(red.similarity_to(pear, "subset"), 0.13945635056250003, places=4)
        self.assertAlmostEqual(pear.similarity_to(non_sweet, "subset"), 0.38164573837037025, places=4)
        self.assertAlmostEqual(non_sweet.similarity_to(pear, "subset"), 0.23419170304545447, places=4)
        self.assertAlmostEqual(non_sweet.similarity_to(red, "subset"), 0.0, places=4)
        self.assertAlmostEqual(red.similarity_to(non_sweet, "subset"), 0.0, places=4)

    def test_similarity_subset_combined_space_properties(self):
        doms = {0:[0],1:[1,2]}       
        cs.init(3, doms)
        c1 = Cuboid([0.00,0.00,0.00],[0.40,0.40,0.40], doms)
        c2 = Cuboid([0.60,0.60,0.60],[1.00,1.00,1.00], doms)
        c3 = Cuboid([float("-inf"),0.40,0.30],[float("inf"),0.40,0.50], {1:[1,2]})
        c4 = Cuboid([0.30,float("-inf"),float("-inf")],[0.40,float("inf"),float("inf")], {0:[0]})
        c5 = Cuboid([1.20,float("-inf"),float("-inf")],[1.40,float("inf"),float("inf")], {0:[0]})
        s1 = Core([c1], doms)
        s2 = Core([c2], doms)
        s3 = Core([c3], {1:[1,2]})
        s4 = Core([c4], {0:[0]})
        s5 = Core([c5], {0:[0]})
        w_a = Weights({0:0.25, 1:1.75}, {0:{0:1}, 1:{1:0.5, 2:0.5}})
        w_b = Weights({1:1}, {1:{1:0.75, 2:0.25}})
        w_c = Weights({0:1}, {0:{0:1}})
        f1 = Concept(s1, 1.0, 1.0, w_a)
        f2 = Concept(s2, 1.0, 1.0, w_a)
        f3 = Concept(s3, 1.0, 1.0, w_b)
        f4 = Concept(s4, 1.0, 1.0, w_c)
        f5 = Concept(s5, 1.0, 1.0, w_c)
        
        self.assertAlmostEqual(f1.similarity_to(f1, method="subset"), 1.0, places=4)
        self.assertAlmostEqual(f1.similarity_to(f2, method="subset"), 0.6036357034764092, places=4)
        self.assertAlmostEqual(f2.similarity_to(f1, method="subset"), 0.6036357034764092, places=4)
        self.assertAlmostEqual(f1.similarity_to(f3, method="subset"), 0.8573456925407419, places=4)
        self.assertAlmostEqual(f3.similarity_to(f1, method="subset"), 0.9784616581266496, places=4)
        self.assertAlmostEqual(f3.similarity_to(f2, method="subset"), 0.8808291122258447, places=4)
        self.assertAlmostEqual(f2.similarity_to(f3, method="subset"), 0.7768064048632629, places=4)
        self.assertAlmostEqual(f4.similarity_to(f1, method="subset"), 1.0, places=4)
        self.assertAlmostEqual(f1.similarity_to(f4, method="subset"), 0.8749999999999999, places=4)
        self.assertAlmostEqual(f5.similarity_to(f1, method="subset"), 0.6093818599999999, places=4)
        self.assertAlmostEqual(f1.similarity_to(f5, method="subset"), 0.5586000383333333, places=4)
        self.assertAlmostEqual(f5.similarity_to(f2, method="subset"), 0.8225794709090909, places=4)
        self.assertAlmostEqual(f2.similarity_to(f5, method="subset"), 0.7540311816666666, places=4)

    
    # between()

    # 'infimum'
    def test_between_infimum(self):
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
        
        self.assertAlmostEqual(f1.between(f1, f1, method="infimum"), 1.0, places=4)
        self.assertAlmostEqual(f3.between(f1, f2, method="infimum"), 1.0, places=4)
        self.assertAlmostEqual(f3.between(f2, f1, method="infimum"), 1.0, places=4)
        self.assertAlmostEqual(f4.between(f1, f2, method="infimum"), 1.0, places=4)
        self.assertAlmostEqual(f4.between(f2, f1, method="infimum"), 1.0, places=4)
        self.assertAlmostEqual(f5.between(f1, f2, method="infimum"), 0.33831631165095066, places=4)
        self.assertAlmostEqual(f5.between(f2, f1, method="infimum"), 0.33831631165095066, places=4)

    def test_between_infimum_fruit(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        dimension_names = ["hue", "round", "sweet"]
        cs.init(3, domains, dimension_names)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        
        c_orange = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
        s_orange = Core([c_orange], domains)
        w_orange = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        orange = Concept(s_orange, 1.0, 15.0, w_orange)
        
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        apple = Concept(s_apple, 1.0, 10.0, w_apple)
        
        c_banana_1 = Cuboid([0.5, 0.1, 0.35], [0.75, 0.30, 0.55], domains)
        c_banana_2 = Cuboid([0.7, 0.1, 0.5], [0.8, 0.3, 0.7], domains)
        c_banana_3 = Cuboid([0.75, 0.1, 0.5], [0.85, 0.3, 1.00], domains)
        s_banana = Core([c_banana_1, c_banana_2, c_banana_3], domains)
        w_banana = Weights({"color":0.75, "shape":1.50, "taste":0.75}, w_dim)
        banana = Concept(s_banana, 1.0, 10.0, w_banana)
        
        self.assertAlmostEqual(banana.between(apple, orange, method='infimum'), 0.0, places=4)
        self.assertAlmostEqual(banana.between(orange, apple, method='infimum'), 0.0, places=4)
        self.assertAlmostEqual(orange.between(apple, banana, method='infimum'), 0.7045928116960077, places=4)
        self.assertAlmostEqual(orange.between(banana, apple, method='infimum'), 0.7045928116960077, places=4)
        self.assertAlmostEqual(apple.between(orange, banana, method='infimum'), 0.0, places=4)
        self.assertAlmostEqual(apple.between(banana, orange, method='infimum'), 0.0, places=4)
        self.assertAlmostEqual(apple.between(apple, banana, method='infimum'), 1.0, places=4)
        self.assertAlmostEqual(apple.between(orange, apple, method='infimum'), 1.0, places=4)

    def test_between_infimum_banana(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        dimension_names = ["hue", "round", "sweet"]
        cs.init(3, domains, dimension_names)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        pear = Concept(s_pear, 1.0, 24.0, w_pear)

        c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
        s_granny_smith = Core([c_granny_smith], domains)
        w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        granny_smith = Concept(s_granny_smith, 1.0, 50.0, w_granny_smith)

        c_banana_1 = Cuboid([0.5, 0.1, 0.35], [0.75, 0.30, 0.55], domains)
        c_banana_2 = Cuboid([0.7, 0.1, 0.5], [0.8, 0.3, 0.7], domains)
        c_banana_3 = Cuboid([0.75, 0.1, 0.5], [0.85, 0.3, 1.00], domains)
        s_banana = Core([c_banana_1, c_banana_2, c_banana_3], domains)
        w_banana = Weights({"color":0.75, "shape":1.50, "taste":0.75}, w_dim)
        banana = Concept(s_banana, 1.0, 20.0, w_banana)

        self.assertAlmostEqual(banana.between(pear, granny_smith, method='infimum'), 0.0, places=4)
        self.assertAlmostEqual(banana.between(granny_smith, pear, method='infimum'), 0.0, places=4)  

      
    def test_between_infimum_pathological(self):
        domains = {0:[0]}
        cs.init(1, domains)
        
        c1 = Cuboid([0],[0], domains)
        c2 = Cuboid([0.5],[0.5], domains)
        c3 = Cuboid([1],[1], domains)
        
        s1 = Core([c1], domains)
        s2 = Core([c2], domains)
        s3 = Core([c3], domains)

        w = Weights({0:1}, {0:{0:1}})        
        
        f1 = Concept(s1, 0.9, 10.0, w)
        f2 = Concept(s3, 0.9, 10.0, w)
        f3 = Concept(s2, 1.0, 10.0, w)
        f4 = Concept(s2, 0.9, 1.0, w)
        
        self.assertAlmostEqual(f3.between(f2, f1, method='infimum'), 0.0, places=4)
        self.assertAlmostEqual(f4.between(f2, f1, method='infimum'), 0.0, places=4)

    # 'integral'
    def test_between_integral(self):
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
      
        self.assertAlmostEqual(f1.between(f1, f1, method="integral"), 1.0, places=4)
        self.assertAlmostEqual(f3.between(f1, f2, method="integral"), 0.99691771963, places=4)
        self.assertAlmostEqual(f3.between(f2, f1, method="integral"), 0.99691771963, places=4)
        self.assertAlmostEqual(f4.between(f1, f2, method="integral"), 1.0, places=4)
        self.assertAlmostEqual(f4.between(f2, f1, method="integral"), 1.0, places=4)
        self.assertAlmostEqual(f5.between(f1, f2, method="integral"), 0.5025296167624573, places=4)
        self.assertAlmostEqual(f5.between(f2, f1, method="integral"), 0.5025296167624573, places=4)

    def test_between_integral_fruit(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        dimension_names = ["hue", "round", "sweet"]
        cs.init(3, domains, dimension_names)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        
        c_orange = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
        s_orange = Core([c_orange], domains)
        w_orange = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        orange = Concept(s_orange, 1.0, 15.0, w_orange)
        
        c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
        c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
        c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
        s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
        w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
        apple = Concept(s_apple, 1.0, 10.0, w_apple)
        
        c_banana_1 = Cuboid([0.5, 0.1, 0.35], [0.75, 0.30, 0.55], domains)
        c_banana_2 = Cuboid([0.7, 0.1, 0.5], [0.8, 0.3, 0.7], domains)
        c_banana_3 = Cuboid([0.75, 0.1, 0.5], [0.85, 0.3, 1.00], domains)
        s_banana = Core([c_banana_1, c_banana_2, c_banana_3], domains)
        w_banana = Weights({"color":0.75, "shape":1.50, "taste":0.75}, w_dim)
        banana = Concept(s_banana, 1.0, 10.0, w_banana)

        self.assertAlmostEqual(banana.between(apple, orange, method='integral'), 0.40303144681997927, places=4)
        self.assertAlmostEqual(banana.between(orange, apple, method='integral'), 0.40303144681997927, places=4)
        self.assertAlmostEqual(orange.between(apple, banana, method='integral'), 0.8478494940748152, places=4)
        self.assertAlmostEqual(orange.between(banana, apple, method='integral'), 0.8478494940748152, places=4)
        self.assertAlmostEqual(apple.between(orange, banana, method='integral'), 0.9022444674696194, places=4)
        self.assertAlmostEqual(apple.between(banana, orange, method='integral'), 0.9022444674696194, places=4)
        self.assertAlmostEqual(apple.between(apple, banana, method='integral'), 1.0, places=4)
        self.assertAlmostEqual(apple.between(orange, apple, method='integral'), 1.0, places=4)

    def test_between_integral_banana(self):
        domains = {"color":[0], "shape":[1], "taste":[2]}
        dimension_names = ["hue", "round", "sweet"]
        cs.init(3, domains, dimension_names)
        w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}
        
        c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
        s_pear = Core([c_pear], domains)
        w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
        pear = Concept(s_pear, 1.0, 24.0, w_pear)

        c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
        s_granny_smith = Core([c_granny_smith], domains)
        w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
        granny_smith = Concept(s_granny_smith, 1.0, 50.0, w_granny_smith)

        c_banana_1 = Cuboid([0.5, 0.1, 0.35], [0.75, 0.30, 0.55], domains)
        c_banana_2 = Cuboid([0.7, 0.1, 0.5], [0.8, 0.3, 0.7], domains)
        c_banana_3 = Cuboid([0.75, 0.1, 0.5], [0.85, 0.3, 1.00], domains)
        s_banana = Core([c_banana_1, c_banana_2, c_banana_3], domains)
        w_banana = Weights({"color":0.75, "shape":1.50, "taste":0.75}, w_dim)
        banana = Concept(s_banana, 1.0, 20.0, w_banana)

        self.assertAlmostEqual(banana.between(pear, granny_smith, method='integral'), 0.435957783784, places=4)
        self.assertAlmostEqual(banana.between(granny_smith, pear, method='integral'), 0.435957783784, places=4)

    def test_between_integral_pathological(self):
        domains = {0:[0]}
        cs.init(1, domains)
        
        c1 = Cuboid([0],[0], domains)
        c2 = Cuboid([0.5],[0.5], domains)
        c3 = Cuboid([1],[1], domains)
        
        s1 = Core([c1], domains)
        s2 = Core([c2], domains)
        s3 = Core([c3], domains)

        w = Weights({0:1}, {0:{0:1}})        
        
        f1 = Concept(s1, 0.9, 10.0, w)
        f2 = Concept(s3, 0.9, 10.0, w)
        f3 = Concept(s2, 1.0, 10.0, w)
        f4 = Concept(s2, 0.9, 1.0, w)
        
        self.assertAlmostEqual(f3.between(f2, f1, method='integral'), 0.9, places=4)
        self.assertAlmostEqual(f4.between(f2, f1, method='integral'), 0.778402196538, places=4)
        self.assertAlmostEqual(f4.between(f2, f1, method='integral', num_alpha_cuts=100), 0.76282936062, places=4)
        self.assertAlmostEqual(f4.between(f2, f1, method='integral', num_alpha_cuts=1000), 0.759107079965, places=4)


    # sample()
    def test_sample_one_dimension_one_cuboid(self):
        random.seed(42)
        doms = {0:[0]}
        cs.init(1, doms)
        s = Core([Cuboid([0.5],[0.7],doms)], doms)
        w = Weights({0:1}, {0:{0:1}})
        f = Concept(s, 1.0, 10.0, w)    
        
        expected_samples = [[0.8205106003073817], [0.4765153714885756], [1.0893800654543875], [0.3473480357360961], 
                            [0.9235408035519259], [0.3946845714812346], [0.6900689273941277], [0.1714756921447516], 
                            [0.697056837377988], [0.440896762692208], [0.6009269677135154], [0.7723419529504013], 
                            [0.2955737618046804], [0.5609078189852715], [0.47843872817402455], [0.6621487706978224], 
                            [0.5021021223612736], [0.5567778328095853], [0.4791356905517389], [0.4459905858967407]]
        samples = f.sample(20)
        self.assertEqual(samples, expected_samples)
        
    def test_sample_one_dimension_one_cuboid_scaled(self):
        random.seed(42)
        doms = {0:[0]}
        cs.init(1, doms)
        s = Core([Cuboid([5],[7],doms)], doms)
        w = Weights({0:1}, {0:{0:1}})
        f = Concept(s, 1.0, 1.0, w)    
        
        expected_samples = [[8.205106003073817], [4.765153714885757], [10.893800654543874], [3.4734803573609607], 
                            [9.23540803551926], [3.946845714812346], [6.900689273941278], [1.7147569214475165], 
                            [6.97056837377988], [4.40896762692208], [6.009269677135154], [7.723419529504014], 
                            [2.955737618046805], [5.609078189852716], [4.784387281740246], [6.621487706978225], 
                            [5.0210212236127365], [5.567778328095852], [4.7913569055173895], [4.459905858967407]]
        samples = f.sample(20)
        self.assertEqual(samples, expected_samples)
        
    def test_sample_two_dimensions_one_cuboid_property(self):
        random.seed(42)
        doms = {0:[0], 1:[1]}
        dom = {0:[0]}
        cs.init(2, doms)
        s = Core([Cuboid([0.5, float("-inf")],[0.7, float("inf")],dom)], dom)
        w = Weights({0:1}, {0:{0:1}})
        f = Concept(s, 1.0, 10.0, w)    
        
        expected_samples = [[0.671077246097072, -1.1182375118372132], [0.7223363669989505, 0.8182873448596939], 
                            [0.8341255198319808, 0.43652402266795276], [0.4349365229310276, 1.658190358962174], 
                            [0.6150663198218392, -1.6363623513048244], [0.47689201330881126, -1.7458891753921715], 
                            [0.5268116788866108, 1.8152637100843205], [0.8197557203077108, 0.43588084575268926], 
                            [0.6480058823075816, -1.997712415488226], [0.5778432024671717, -1.7231499261264656], 
                            [0.6787669258743846, -0.9397734842397636], [0.47843872817402455, -1.1528071782316718], 
                            [0.6277970899463485, -1.5159832165269371], [0.7123582792556478, -0.10931589475282344], 
                            [0.4909539247388911, -0.3056855079203169], [0.5187297023218571, -0.31247344066238325], 
                            [0.5772907067965353, -1.1450108032032733], [0.6882004507621521, 0.873633101185304], 
                            [0.6667338652830263, 0.9919022415162564], [0.4722500795674033, 0.3346891571648989]]
        samples = f.sample(20)
        self.assertEqual(samples, expected_samples)
        

unittest.main()