# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:25:20 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
from cs.cuboid import Cuboid

class TestCuboid(unittest.TestCase):

    # _check()
    def test_check_true(self):
        c = Cuboid([1, 2, 3], [2, 3, 4])
        self.assertTrue(c._check())
    
    def test_check_true_same(self):    
        c = Cuboid([1, 2, 3], [1, 2, 3])
        self.assertTrue(c._check())
    
    def test_check_false(self):
        c = Cuboid( [1,2,3], [2,3,4])
        self.assertFalse(c._check([1,2,3], [2,3,2]))
    
    def test_check_empty(self):
        c = Cuboid([1,2,3], [2,3,4])
        with self.assertRaises(Exception):
            c._check([],[])
    
    def test_check_empty_init(self):
        with self.assertRaises(Exception):
            Cuboid([], [])
    
    def test_check_init_different_length(self):
        with self.assertRaises(Exception):
            Cuboid([1,2,3],[4,5])
    
    def test_check_init_different_length_rev(self):
        with self.assertRaises(Exception):
            Cuboid([1,2],[4,5,6])
    
    # contains()
    def test_contains_true(self):
        c = Cuboid([1,2,3],[7,8,9])
        self.assertTrue(c.contains([4,5,6]))
    
    def test_contains_false(self):
        c = Cuboid([1,2,3],[7,8,9])
        self.assertFalse(c.contains([0,5,6]))
    
    def test_contains_point_too_short(self):
        c = Cuboid([1,2,3],[7,8,9])
        with self.assertRaises(Exception):
            c.contains([0,5])
    
    def test_contains_point_too_long(self):
        c = Cuboid([1,2,3],[7,8,9])
        with self.assertRaises(Exception):
            c.contains([0,5,6,3])
     
    # _find_closest_point()
    def test_find_closest_point_too_short(self):
        c = Cuboid([1,2,3],[7,8,9])
        with self.assertRaises(Exception):
            c._find_closest_point([0,5])
    
    def test_find_closest_point_too_long(self):
        c = Cuboid([1,2,3],[7,8,9])
        with self.assertRaises(Exception):
            c._find_closest_point([0,5,6,3])
    
    def test_find_closest_point_inside(self):
        c = Cuboid([1,2,3],[7,8,9])
        p = [4,5,6]
        self.assertEqual(p, c._find_closest_point(p))
    
    def test_find_closest_point_one_difference(self):
        c = Cuboid([1,2,3],[7,8,9])
        p = [4,5,10]
        self.assertEqual(c._find_closest_point(p), [4,5,9])
    
    def test_find_closest_point_two_differences(self):
        c = Cuboid([1,2,3],[7,8,9])
        p = [12,-2,7]
        self.assertEqual(c._find_closest_point(p), [7,2,7])


unittest.main()