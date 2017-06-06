# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 13:00:00 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
from cs.weights import Weights

class TestWeights(unittest.TestCase):

    # constructor
    def test_constructor_fine_uniform(self):
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}

        w = Weights(dom, dim)        
        
        self.assertEqual(w.domain_weights, dom)
        self.assertEqual(w.dimension_weights, dim)
    
    def test_constructor_fine_change(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}

        w = Weights(dom, dim)   
        dom_new = {0:(4.0/3), 1:(2.0/3)}
        dim_new = {0:{0:0.5, 1:0.5}, 1:{2:0.6, 3:0.4}}
        
        self.assertEqual(w.domain_weights, dom_new)
        self.assertEqual(w.dimension_weights, dim_new)
        
    # _check()    
    def test_check_true(self):
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}

        w = Weights(dom, dim)
        
        self.assertTrue(w._check())
        self.assertTrue(w._check(dom,dim))
    
    def test_check_false(self):
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}

        w = Weights(dom, dim)
        
        self.assertTrue(w._check())
        self.assertFalse(w._check(dom,dim))
    
unittest.main()