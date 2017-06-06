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


unittest.main()