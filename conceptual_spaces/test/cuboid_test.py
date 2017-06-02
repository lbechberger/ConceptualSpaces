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
    
    # __eq__()
    def test_eq_identity(self):
         c = Cuboid([1,2,3],[7,8,9])
         self.assertEqual(c,c)

    def test_eq_same(self):
         c1 = Cuboid([1,2,3],[7,8,9])
         c2 = Cuboid([1,2,3],[7,8,9])
         self.assertEqual(c1,c2)

    def test_eq_different_cuboids(self):
         c1 = Cuboid([1,2,3],[7,8,9])
         c2 = Cuboid([1,2,3],[7,8,8])
         self.assertFalse(c1 == c2)

    def test_eq_different_types(self):
         c1 = Cuboid([1,2,3],[7,8,9])
         x = 42
         self.assertFalse(c1 == x)

    def test_eq_None(self):
         c1 = Cuboid([1,2,3],[7,8,9])
         x = None
         self.assertFalse(c1 == x)

    # __ne__()    
    def test_ne_identity(self):
         c = Cuboid([1,2,3],[7,8,9])
         self.assertFalse(c != c)

    def test_ne_same(self):
         c1 = Cuboid([1,2,3],[7,8,9])
         c2 = Cuboid([1,2,3],[7,8,9])
         self.assertFalse(c1 != c2)

    def test_ne_different_cuboids(self):
         c1 = Cuboid([1,2,3],[7,8,9])
         c2 = Cuboid([1,2,3],[7,8,8])
         self.assertTrue(c1 != c2)

    def test_ne_different_types(self):
         c1 = Cuboid([1,2,3],[7,8,9])
         x = 42
         self.assertTrue(c1 != x)

    def test_ne_None(self):
         c1 = Cuboid([1,2,3],[7,8,9])
         x = None
         self.assertTrue(c1 != x)
    
    # intersect()
    def test_intersect_identity(self):
        c = Cuboid([1,2,3],[7,8,9])
        self.assertEqual(c.intersect(c),c)

    def test_intersect_empty(self):
        c1 = Cuboid([0,0,0],[1,1,1])
        c2 = Cuboid([2,2,2],[3,3,3])
        self.assertEqual(c1.intersect(c2), None)
    
    def test_intersect_3d(self):
        c1 = Cuboid([0,0,0],[2,2,2])
        c2 = Cuboid([1,1,1],[3,3,3])
        c3 = Cuboid([1,1,1],[2,2,2])
        self.assertEqual(c1.intersect(c2), c3)

    def test_intersect_2d(self):
        c1 = Cuboid([0,0,0],[2,2,2])
        c2 = Cuboid([2,1,1],[3,3,3])
        c3 = Cuboid([2,1,1],[2,2,2])
        self.assertEqual(c1.intersect(c2), c3)

    def test_intersect_1d(self):
        c1 = Cuboid([0,0,0],[2,2,2])
        c2 = Cuboid([2,2,1],[3,3,3])
        c3 = Cuboid([2,2,1],[2,2,2])
        self.assertEqual(c1.intersect(c2), c3)

    def test_intersect_0d(self):
        c1 = Cuboid([0,0,0],[2,2,2])
        c2 = Cuboid([2,2,2],[3,3,3])
        c3 = Cuboid([2,2,2],[2,2,2])
        self.assertEqual(c1.intersect(c2), c3)

    def test_intersect_commutative(self):
        c1 = Cuboid([0,0,0],[2,2,2])
        c2 = Cuboid([1,1,1],[3,3,3])
        self.assertEqual(c1.intersect(c2), c2.intersect(c1))

    def test_intersect_other_object(self):
        c1 = Cuboid([0,0,0],[2,2,2])
        x = 42
        with self.assertRaises(Exception):
            c1.intersect(x)
    
    def test_intersect_different_length(self):
        c1 = Cuboid([0,0,0],[2,2,2])
        c2 = Cuboid([0,0,0,0],[2,2,2,2])
        with self.assertRaises(Exception):
            c1.intersect(c2)

unittest.main()