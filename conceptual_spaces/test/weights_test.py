# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:00:00 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
from cs.weights import Weights, check

class TestWeights(unittest.TestCase):

    # constructor
    def test_constructor_fine_uniform(self):
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}

        w = Weights(dom, dim)        
        
        self.assertEqual(w._domain_weights, dom)
        self.assertEqual(w._dimension_weights, dim)
    
    def test_constructor_fine_change(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}

        w = Weights(dom, dim)   
        dom_new = {0:4.0/3, 1:2.0/3}
        dim_new = {0:{0:0.5, 1:0.5}, 1:{2:0.6, 3:0.4}}
        
        self.assertEqual(w._domain_weights, dom_new)
        self.assertEqual(w._dimension_weights, dim_new)
        
    # _check()    
    def test_check_true(self):
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        self.assertTrue(check(dom,dim))
    
    def test_check_false(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        self.assertFalse(check(dom,dim))
    
    # __eq()__ and __ne()__
    def test_eq_ne_identity(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}

        w = Weights(dom, dim)
        self.assertTrue(w == w)
        self.assertFalse(w != w)

    def test_eq_ne_shallow_copy(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}

        w = Weights(dom, dim)
        w2 = Weights(dom, dim)
        self.assertTrue(w == w2)
        self.assertFalse(w != w2)

    def test_eq_ne_deep_copy(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        dom2 = {0:(4.0/3), 1:(2.0/3)}
        dim2 = {0:{0:0.5, 1:0.5}, 1:{2:0.6, 3:0.4}}
        w2 = Weights(dom2, dim2)
        self.assertTrue(w == w2)
        self.assertFalse(w != w2)

    def test_eq_ne_different_weights(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        dom2 = {0:1, 1:1}        
        dim2 = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w2 = Weights(dom2, dim2)
        self.assertTrue(w != w2)
        self.assertFalse(w == w2)
    
    # merge_with()
    def test_merge_identity(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}

        w = Weights(dom, dim)
        self.assertEqual(w.merge_with(w), w)
    
    def test_merge_fifty_fifty_same_doms(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)

        dom2 = {0:1, 1:1}        
        dim2 = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w2 = Weights(dom2, dim2)
        
        dom_res = {0:(7.0/6.0), 1:(5.0/6.0)}
        dim_res = {0:{0:0.5, 1:0.5}, 1:{2:0.55, 3:0.45}}
        w_res = Weights(dom_res, dim_res)

        self.assertEqual(w.merge_with(w2), w_res)
        self.assertEqual(w.merge_with(w2), w2.merge_with(w))        

    def test_merge_three_to_one_same_doms(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)

        dom2 = {0:1, 1:1}        
        dim2 = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w2 = Weights(dom2, dim2)
        
        dom_res = {0:1.25, 1:0.75}
        dim_res = {0:{0:0.5, 1:0.5}, 1:{2:0.575, 3:0.42500000000000004}} # weird rounding error in python!
        w_res = Weights(dom_res, dim_res)
        
        self.assertEqual(w.merge_with(w2, 0.75, 0.75), w_res)
        self.assertEqual(w.merge_with(w2, 0.75, 0.75), w2.merge_with(w, 0.25, 0.25))        

    def test_merge_overlapping_doms(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)

        dom2 = {1:1, 2:2}        
        dim2 = {1:{2:0.5, 3:0.5}, 2:{4:1, 5:1, 6:2}}
        w2 = Weights(dom2, dim2)
        
        dom_res = {0:(4.0)/3, 1:(2.0)/3, 2:4.0/3}
        dim_res = {0:{0:0.5, 1:0.5}, 1:{2:0.55, 3:0.45}, 2:{4:0.25, 5:0.25, 6:0.5}}
        w_res = Weights(dom_res, dim_res)

        self.assertEqual(w.merge_with(w2), w_res)
        self.assertEqual(w.merge_with(w2), w2.merge_with(w))        

    def test_merge_different_doms(self):
        dom = {1:1}        
        dim = {1:{2:3, 3:2.0}}
        w = Weights(dom, dim)

        dom2 = {0:1}        
        dim2 = {0:{0:0.5, 1:0.5}}
        w2 = Weights(dom2, dim2)
        
        dom_res = {0:1, 1:1}
        dim_res = {0:{0:0.5, 1:0.5}, 1:{2:0.6, 3:0.4}}
        w_res = Weights(dom_res, dim_res)

        self.assertEqual(w.merge_with(w2), w_res)
        self.assertEqual(w.merge_with(w2), w2.merge_with(w))        
        
    # project_onto()
    def test_project_2_dom_to_1(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        dom_res = {0:1}
        dim_res = {0:{0:1, 1:1}}
        w_res = Weights(dom_res, dim_res)
        
        self.assertEqual(w.project_onto({0:[0,1]}), w_res)
 
    def test_project_3_dom_to_2(self):
        dom = {0:2, 1:1, 2:3}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}, 2:{4:1}}
        w = Weights(dom, dim)
        
        dom_res = {0:2, 2:3}
        dim_res = {0:{0:1, 1:1}, 2:{4:1}}
        w_res = Weights(dom_res, dim_res)
        
        self.assertEqual(w.project_onto({0:[0,1], 2:[4]}), w_res)
    
unittest.main()