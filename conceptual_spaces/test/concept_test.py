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

class TestCore(unittest.TestCase):

    # constructor
    def test_constructor_fine(self):
        s = Core([Cuboid([1,2],[3,4])])
        f = Concept(s, 1.0, 2.0, None)        
        
        self.assertEqual(f._core, s)
        self.assertEqual(f._mu, 1.0)
        self.assertEqual(f._c, 2.0)
        self.assertEqual(f._weights, None)
    
    def test_constructor_wrong_core(self):
        with self.assertRaises(Exception):        
            Concept(42, 1.0, 2.0, None)        
        
    def test_constructor_wrong_mu(self):
        s = Core([Cuboid([1,2],[3,4])])
        with self.assertRaises(Exception):        
            Concept(s, 0.0, 2.0, None)        
    
    def test_constructor_wrong_c(self):
        s = Core([Cuboid([1,2],[3,4])])
        with self.assertRaises(Exception):        
            Concept(s, 1.0, -1.0, None)        

    
unittest.main()