# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 10:51:06 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
from cs.core import Core
from cs.cuboid import Cuboid

class TestCore(unittest.TestCase):
    
    # constructor
    def test_constructor_no_arg(self):
        with self.assertRaises(Exception):
            Core([])
    
    def test_constructor_correct_arg(self):
        c1 = Cuboid([1,2,3],[4,5,6])
        c2 = Cuboid([2,3,4],[5,6,7])
        l = [c1, c2]
        s = Core(l)
        self.assertEqual(s._cuboids, l)
    
    def test_constructor_no_list(self):
        with self.assertRaises(Exception):
            Core(42)
    
    def test_constructor_no_cuboid_list(self):
        with self.assertRaises(Exception):
            Core([Cuboid([1,2],[3,4]), 42, "test"])
    
    def test_constructor_nonintersecting(self):
        c1 = Cuboid([1,2,3],[4,5,6])
        c2 = Cuboid([0,0,0],[1,1,1])
        l = [c1, c2]
        with self.assertRaises(Exception):
            Core(l)

    def test_constructor_different_size(self):
        c1 = Cuboid([1,2,3],[4,5,6])
        c2 = Cuboid([0,0],[1,1])
        l = [c1, c2]
        with self.assertRaises(Exception):
            Core(l)

    # _check
    def test_check_true(self):
        c1 = Cuboid([1,2,3],[4,5,6])
        c2 = Cuboid([2,3,4],[5,6,7])
        c3 = Cuboid([2,2,2],[12.4,12.5,12.6])
        l = [c1, c2, c3]
        s = Core(l)
        self.assertTrue(s._check())
    
    def test_check_false(self):
        c1 = Cuboid([1,2,3],[4,5,6])
        c2 = Cuboid([0,0,0],[1,1,1])
        c3 = Cuboid([1,1,1],[2,3,4])
        l = [c1, c2, c3]
        s = Core([c1])
        self.assertFalse(s._check(l))
    
    # add_cuboid
    def test_add_cuboid_true(self):
        c1 = Cuboid([1,2,3],[4,5,6])
        c2 = Cuboid([2,3,4],[5,6,7])
        c3 = Cuboid([2,2,2],[12.4,12.5,12.6])
        l = [c1]
        s = Core(l)
        self.assertTrue(s.add_cuboid(c2))
        self.assertEqual(s._cuboids, [c1, c2])
        self.assertTrue(s.add_cuboid(c3))
        self.assertEqual(s._cuboids, [c1, c2, c3])

    def test_add_cuboid_false(self):
        c1 = Cuboid([1,2,3],[4,5,6])
        c2 = Cuboid([0,0,0],[1,1,1])
        c3 = Cuboid([1,1,1],[2,3,4])
        l = [c1]
        s = Core(l)
        self.assertFalse(s.add_cuboid(c2))
        self.assertEqual(s._cuboids, [c1])
        self.assertTrue(s.add_cuboid(c3))
        self.assertEqual(s._cuboids, [c1, c3])

    def test_add_cuboid_no_cuboid(self):
        c1 = Cuboid([1,2,3],[4,5,6])
        l = [c1]
        s = Core(l)
        with self.assertRaises(Exception):
            s.add_cuboid(42)
        self.assertEqual(s._cuboids, [c1])

    # find_closest_point_candidates
    def test_find_closest_point_candidates_one_cuboid(self):
        c = Cuboid([1,2,3],[7,8,9])
        s = Core([c])
        p = [12,-2,7]
        self.assertEqual(s.find_closest_point_candidates(p), [[7,2,7]])

    def test_find_closest_point_candidates_two_cuboids(self):
        c1 = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([4,5,6],[7,7,7])
        s = Core([c1, c2])
        p = [12,-2,8]
        self.assertEqual(s.find_closest_point_candidates(p), [[7,2,8],[7,5,7]])

    # __eq__(), __ne__()
    def test_eq_ne_identity(self):
        c = Cuboid([1,2,3],[7,8,9])
        s = Core([c])
        self.assertTrue(s == s)
        self.assertFalse(s != s)

    def test_eq_ne_no_core(self):
        c = Cuboid([1,2,3],[7,8,9])
        s = Core([c])
        self.assertTrue(s != c)
        self.assertFalse(s == c)

    def test_eq_ne_shallow_copy(self):
        c = Cuboid([1,2,3],[7,8,9])
        s = Core([c])
        s2 = Core([c])
        self.assertTrue(s == s2)
        self.assertFalse(s != s2)

    def test_eq_ne_deep_copy(self):
        c = Cuboid([1,2,3],[7,8,9])
        s = Core([c])
        c2 = Cuboid([1,2,3],[7,8,9])
        s2 = Core([c2])
        self.assertTrue(s == s2)
        self.assertFalse(s != s2)

    def test_eq_ne_reversed_cuboid_order(self):
        c = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([6,5,4],[9,8,7])
        s = Core([c, c2])
        s2 = Core([c2, c])
        self.assertTrue(s == s2)
        self.assertFalse(s != s2)

    def test_eq_ne_different_cores(self):
        c = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([6,5,4],[9,8,7])
        s = Core([c])
        s2 = Core([c2])
        self.assertTrue(s != s2)
        self.assertFalse(s == s2)

    # unify()
    def test_unify_no_core(self):
        c = Cuboid([1,2,3],[7,8,9])
        s = Core([c])
        with self.assertRaises(Exception):
            s.unify(42)
    
    def test_unify_no_repair(self):
        c1 = Cuboid([1,2,3],[7,8,9])
        c2 = Cuboid([4,5,6],[7,7,7])
        s1 = Core([c1])
        s2 = Core([c2])
        s_result = Core([c1, c2])
        self.assertEqual(s1.unify(s2), s_result)
        self.assertEqual(s1.unify(s2), s2.unify(s1))
    
    def test_unify_repair(self):
        c1 = Cuboid([1,2,3],[2,3,4])
        c2 = Cuboid([3,4,5],[7,7,7])
        s1 = Core([c1])
        s2 = Core([c2])
        c1_result = Cuboid([1,2,3],[3.25,4,4.75])
        c2_result = Cuboid([3,4,4.75],[7,7,7])
        s_result = Core([c1_result, c2_result])
        self.assertEqual(s1.unify(s2), s_result)
        self.assertEqual(s1.unify(s2), s2.unify(s1))
   

unittest.main()