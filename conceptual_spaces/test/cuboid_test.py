# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 14:25:20 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
from cs.cuboid import Cuboid
import cs.cs as cs

class TestCuboid(unittest.TestCase):

    def _create_cs(self):
        cs.init(3, {0:[0,1,2]})

    # _check()
    def test_check_true(self):
        self._create_cs()
        c = Cuboid([1, 2, 3], [2, 3, 4], {0:[0,1,2]})
        self.assertTrue(c._check())
    
    def test_check_true_same(self):    
        self._create_cs()
        c = Cuboid([1, 2, 3], [1, 2, 3], {0:[0,1,2]})
        self.assertTrue(c._check())
    
    def test_check_false(self):
        self._create_cs()
        c = Cuboid( [1,2,3], [2,3,4], {0:[0,1,2]})
        self.assertFalse(c._check([1,2,3], [2,3,2], {0:[0,1,2]}))
    
    def test_check_empty(self):
        self._create_cs()
        c = Cuboid([1,2,3], [2,3,4], {0:[0,1,2]})
        self.assertFalse(c._check([],[], {0:[0,1,2]}))
    
    # constructor()
    def test_init_empty_init(self):
        self._create_cs()
        with self.assertRaises(Exception):
            Cuboid([], [], {})
    
    def test_init_different_length(self):
        self._create_cs()
        with self.assertRaises(Exception):
            Cuboid([1,2,3],[4,5], {0:[0,1,2]})
    
    def test_init_different_length_rev(self):
        self._create_cs()
        with self.assertRaises(Exception):
            Cuboid([1,2],[4,5,6], {0:[0,1,2]})
    
    def test_init_correct(self):
        cs.init(6, {0:[0,1,3,4], 1:[2], 2:[5]})
        c = Cuboid([1,2,float("-inf"),4,5,float("-inf")],[6,7,float("inf"),8,9,float("inf")], {0:[0,1,3,4]})
        self.assertEqual(c._domains, {0:[0,1,3,4]})

    def test_init_correct_two_domains(self):
        cs.init(6, {0:[0,1], 1:[3,4], 2:[2,5]})
        c = Cuboid([1,2,float("-inf"),4,5,float("-inf")],[6,7,float("inf"),8,9,float("inf")], {0:[0,1], 1:[3,4]})
        self.assertEqual(c._domains, {0:[0,1], 1:[3,4]})

    def test_init_incorrect_domain_inf(self):
        cs.init(6, {0:[0,1,3,4], 1:[2], 2:[5]})
        with self.assertRaises(Exception):
            Cuboid([1,2,float("-inf"),4,5,float("-inf")],[6,7,float("inf"),8,9,float("inf")], {0:[0,1,3,5]})

    def test_init_incorrect_domain(self):
        cs.init(6, {0:[0,1,3,4], 1:[2], 2:[5]})
        with self.assertRaises(Exception):
            Cuboid([1,2,float("-inf"),4,5,float("-inf")],[6,7,float("inf"),8,9,float("inf")], {0:[0,1], 3:[3,4]})

    def test_init_unmatching_inf(self):
        cs.init(6, {0:[0,1,3,4], 1:[2], 2:[5]})
        with self.assertRaises(Exception):
            Cuboid([1,2,float("-inf"),4,5,0],[6,7,float("inf"),8,9,float("inf")],{0:[0,1,3,4]})
    
    # contains()
    def test_contains_true(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        self.assertTrue(c.contains([4,5,6]))
    
    def test_contains_false(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        self.assertFalse(c.contains([0,5,6]))
    
    def test_contains_point_too_short(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        with self.assertRaises(Exception):
            c.contains([0,5])
    
    def test_contains_point_too_long(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        with self.assertRaises(Exception):
            c.contains([0,5,6,3])
    
    def test_contains_true_infinity(self):
        cs.init(3, {0:[0,1], 1:[2]})        
        c = Cuboid([1,2,float("-inf")], [4,5,float("inf")],{0:[0,1]})
        self.assertTrue(c.contains([2,3,5]))
     
    # find_closest_point()
    def test_find_closest_point_too_short(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        with self.assertRaises(Exception):
            c.find_closest_point([0,5])
    
    def test_find_closest_point_too_long(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        with self.assertRaises(Exception):
            c.find_closest_point([0,5,6,3])
    
    def test_find_closest_point_inside(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        p = [4,5,6]
        self.assertEqual(p, c.find_closest_point(p))
    
    def test_find_closest_point_one_difference(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        p = [4,5,10]
        self.assertEqual(c.find_closest_point(p), [4,5,9])
    
    def test_find_closest_point_two_differences(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        p = [12,-2,7]
        self.assertEqual(c.find_closest_point(p), [7,2,7])

    def test_find_closest_point_infinity(self):
        cs.init(3, {0:[0,1], 1:[2]})
        c = Cuboid([1,2,float("-inf")],[7,8,float("inf")], {0:[0,1]})
        p = [4,10,3]
        self.assertEqual(c.find_closest_point(p), [4,8,3])
    
    # __eq__() and __ne__()
    def test_eq_ne_identity(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9], {0:[0,1,2]})
        self.assertEqual(c,c)
        self.assertFalse(c != c)

    def test_eq_ne_same(self):
        self._create_cs()
        c1 = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        c2 = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        self.assertEqual(c1,c2)
        self.assertFalse(c1 != c2)

    def test_eq_ne_different_cuboids(self):
        self._create_cs()
        c1 = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        c2 = Cuboid([1,2,3],[7,8,8],{0:[0,1,2]})
        self.assertFalse(c1 == c2)
        self.assertTrue(c1 != c2)

    def test_eq_ne_different_types(self):
        self._create_cs()
        c1 = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        x = 42
        self.assertFalse(c1 == x)
        self.assertTrue(c1 != x)

    def test_eq_ne_None(self):
        self._create_cs()
        c1 = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        x = None
        self.assertFalse(c1 == x)
        self.assertTrue(c1 != x)

    # intersect()
    def test_intersect_identity(self):
        self._create_cs()
        c = Cuboid([1,2,3],[7,8,9],{0:[0,1,2]})
        self.assertEqual(c.intersect(c),c)

    def test_intersect_empty(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[1,1,1],{0:[0,1,2]})
        c2 = Cuboid([2,2,2],[3,3,3],{0:[0,1,2]})
        self.assertEqual(c1.intersect(c2), None)
    
    def test_intersect_3d(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        c2 = Cuboid([1,1,1],[3,3,3],{0:[0,1,2]})
        c3 = Cuboid([1,1,1],[2,2,2],{0:[0,1,2]})
        self.assertEqual(c1.intersect(c2), c3)

    def test_intersect_2d(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        c2 = Cuboid([2,1,1],[3,3,3],{0:[0,1,2]})
        c3 = Cuboid([2,1,1],[2,2,2],{0:[0,1,2]})
        self.assertEqual(c1.intersect(c2), c3)

    def test_intersect_1d(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        c2 = Cuboid([2,2,1],[3,3,3],{0:[0,1,2]})
        c3 = Cuboid([2,2,1],[2,2,2],{0:[0,1,2]})
        self.assertEqual(c1.intersect(c2), c3)

    def test_intersect_0d(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        c2 = Cuboid([2,2,2],[3,3,3],{0:[0,1,2]})
        c3 = Cuboid([2,2,2],[2,2,2],{0:[0,1,2]})
        self.assertEqual(c1.intersect(c2), c3)

    def test_intersect_commutative(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        c2 = Cuboid([1,1,1],[3,3,3],{0:[0,1,2]})
        self.assertEqual(c1.intersect(c2), c2.intersect(c1))

    def test_intersect_other_object(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        x = 42
        with self.assertRaises(Exception):
            c1.intersect(x)
    
    def test_intersect_one_infinity(self):
        cs.init(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([0,0,float("-inf")],[2,2,float("inf")],{0:[0,1]})
        c2 = Cuboid([2,1,1],[3,3,3],{0:[0,1],1:[2]})
        c3 = Cuboid([2,1,1],[2,2,3],{0:[0,1],1:[2]})
        self.assertEqual(c1.intersect(c2), c3)
        self.assertEqual(c1.intersect(c2), c2.intersect(c1))

    def test_intersect_two_infinity_same(self):
        cs.init(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([0,0,float("-inf")],[2,2,float("inf")],{0:[0,1]})
        c2 = Cuboid([2,1,float("-inf")],[3,3,float("inf")],{0:[0,1]})
        c3 = Cuboid([2,1,float("-inf")],[2,2,float("inf")],{0:[0,1]})
        self.assertEqual(c1.intersect(c2), c3)
        self.assertEqual(c1.intersect(c2), c2.intersect(c1))

    def test_intersect_two_infinity_different(self):
        cs.init(3, {0:[0], 1:[1], 2:[2]})
        c1 = Cuboid([0,0,float("-inf")],[2,2,float("inf")],{0:[0], 1:[1]})
        c2 = Cuboid([2,float("-inf"),1],[3,float("inf"),3],{0:[0], 2:[2]})
        c3 = Cuboid([2,0,1],[2,2,3],{0:[0], 1:[1], 2:[2]})
        self.assertEqual(c1.intersect(c2), c3)
        self.assertEqual(c1.intersect(c2), c2.intersect(c1))

    # project()
    def test_project_illegal_domains_subdomain(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        with self.assertRaises(Exception):
            c1.project({0:[1,2]})
    
    def test_project_illegal_domains_other_domain_name(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        with self.assertRaises(Exception):
            c1.project({1:[0,1,2]})

    def test_project_identical_domains(self):
        self._create_cs()
        c1 = Cuboid([0,0,0],[2,2,2],{0:[0,1,2]})
        self.assertEqual(c1.project({0:[0,1,2]}), c1)
    
    def test_project_correct(self):
        cs.init(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([0,1,2],[3,4,5],{0:[0,1], 1:[2]})
        c_res1 = Cuboid([0,1,float("-inf")],[3,4,float("inf")],{0:[0,1]})
        c_res2 = Cuboid([float("-inf"),float("-inf"),2],[float("inf"),float("inf"),5],{1:[2]})
        self.assertEqual(c1.project({0:[0,1]}), c_res1)
        self.assertEqual(c1.project({1:[2]}), c_res2)
        
    # get_closest_points()
    def test_get_closest_points_no_overlap_same_domains(self):
        self._create_cs()
        c1 = Cuboid([0,1,2],[1,2,3], {0:[0,1,2]})
        c2 = Cuboid([2,3,4],[3,4,5], {0:[0,1,2]})
        a_res = [[1,1],[2,2],[3,3]]
        b_res = [[2,2],[3,3],[4,4]]
        a, b = c1.get_closest_points(c2)
        b2, a2 = c2.get_closest_points(c1)
        self.assertEqual(a, a_res)
        self.assertEqual(b, b_res)
        self.assertEqual(a, a2)
        self.assertEqual(b, b2)

    def test_get_closest_points_two_overlaps_same_domains(self):
        self._create_cs()
        c1 = Cuboid([0,1,2],[1,2,3], {0:[0,1,2]})
        c2 = Cuboid([1,1,4],[3,4,5], {0:[0,1,2]})
        a_res = [[1,1],[1,2],[3,3]]
        b_res = [[1,1],[1,2],[4,4]]
        a, b = c1.get_closest_points(c2)
        b2, a2 = c2.get_closest_points(c1)
        self.assertEqual(a, a_res)
        self.assertEqual(b, b_res)
        self.assertEqual(a, a2)
        self.assertEqual(b, b2)

    def test_get_closest_points_othorgonal_domains(self):
        cs.init(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([0,1,float("-inf")],[1,2,float("inf")], {0:[0,1]})
        c2 = Cuboid([float("-inf"),float("-inf"),4],[float("inf"),float("inf"),5], {1:[2]})
        a_res = [[0,1],[1,2],[4,5]]
        b_res = [[0,1],[1,2],[4,5]]
        a, b = c1.get_closest_points(c2)
        b2, a2 = c2.get_closest_points(c1)
        self.assertEqual(a, a_res)
        self.assertEqual(b, b_res)
        self.assertEqual(a, a2)
        self.assertEqual(b, b2)

    def test_get_closest_points_subdomains(self):
        cs.init(3, {0:[0,1], 1:[2]})
        c1 = Cuboid([0,1,float("-inf")],[1,2,float("inf")], {0:[0,1]})
        c2 = Cuboid([2,1,4],[3,4,5], {0:[0,1],1:[2]})
        a_res = [[1,1],[1,2],[4,5]]
        b_res = [[2,2],[1,2],[4,5]]
        a, b = c1.get_closest_points(c2)
        b2, a2 = c2.get_closest_points(c1)
        self.assertEqual(a, a_res)
        self.assertEqual(b, b_res)
        self.assertEqual(a, a2)
        self.assertEqual(b, b2)

unittest.main()