# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 12:32:16 2017

@author: lbechberger
"""

import unittest
import sys
sys.path.append("..")
import cs.cs as space
from cs.weights import Weights
from cs.core import Core
from cs.concept import Concept
from cs.cuboid import Cuboid
from math import sqrt

class TestConceptualSpace(unittest.TestCase):

    # constructor()
    def test_constructor_fine(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        

        space.init(n, domains)        
        
        self.assertEqual(space._n_dim, n)
        self.assertEqual(space._domains, domains)
    
    def test_constructor_negative_n(self):
        n = -1        
        domains = {0:[0,1], 1:[2,3]}        
        with self.assertRaises(Exception):        
            space.init(n, domains)
        
    def test_constructor_overlapping_domains(self):
        n = 4        
        domains = {0:[0,1,2], 1:[2,3]}        
        with self.assertRaises(Exception):        
            space.init(n, domains)
    
    def test_constructor_illegal_dimensions(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3,4]}        
        with self.assertRaises(Exception):        
            space.init(n, domains)
            
    def test_constructor_missing_dimensions(self):
        n = 4        
        domains = {0:[0,1], 1:[3]}        
        with self.assertRaises(Exception):        
            space.init(n, domains)

    def test_constructor_empty_domain(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3], 2:[]}        
        with self.assertRaises(Exception):        
            space.init(n, domains)
 
    def test_constructor_no_dim_names(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        res_names = ["dim_0", "dim_1", "dim_2", "dim_3"]

        space.init(n, domains)        
        self.assertEqual(space._dim_names, res_names)

    def test_constructor_dim_names(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        dim_names = ["hue", "saturation", "width", "height"]

        space.init(n, domains, dim_names)        
        self.assertEqual(space._dim_names, dim_names)

    def test_constructor_dim_names_fail(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        dim_names = ["hue", "saturation", "width"]

        with self.assertRaises(Exception):
            space.init(n, domains, dim_names)        
 
    # distance()
    def test_distance_illegal_point(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        space.init(n, domains)
        
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [5,6,7]
        with self.assertRaises(Exception):
            space.distance(x,y,w)
    
    def test_distance_unit_diff_identically_weighted(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        space.init(n, domains)
        
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [2,3,2,3]   # distance of 1 wrt each coordinate
        self.assertEqual(space.distance(x,y,w), 2.0)
        self.assertEqual(space.distance(x,y,w), space.distance(y,x,w))

    def test_distance_unit_diff_differently_weighted(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        space.init(n, domains)
        
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [2,3,2,3]   # distance of 1 wrt each coordinate
        self.assertEqual(space.distance(x,y,w), 2.0)
        self.assertEqual(space.distance(x,y,w), space.distance(y,x,w))
        
    def test_distance_other_diff_identically_weighted(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        space.init(n, domains)
        
        dom = {0:1, 1:1}        
        dim = {0:{0:0.5, 1:0.5}, 1:{2:0.5, 3:0.5}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [2,0,2,2]   # difference: 1 2 1 2
        self.assertEqual(space.distance(x,y,w), sqrt(0.5*1 + 0.5*4) + sqrt(0.5*1 + 0.5*4))
        self.assertEqual(space.distance(x,y,w), space.distance(y,x,w))

    def test_distance_other_diff_differently_weighted(self):
        n = 4        
        domains = {0:[0,1], 1:[2,3]}        
        space.init(n, domains)
        
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)

        x = [1,2,3,4]
        y = [2,0,2,2]   # difference: 1 2 1 2
        self.assertEqual(space.distance(x,y,w), (4.0/3)*sqrt(0.5*1+0.5*4) + (2.0/3)*sqrt(0.6*1 + 0.4*4))
        self.assertEqual(space.distance(x,y,w), space.distance(y,x,w))
    
    # add_concept()
    def test_add_concept_failure(self):
        space.init(4, {0:[0,1], 1:[2,3]})
        
        with self.assertRaises(Exception):
            space.add_concept(42, 1337)

    def test_add_concept_correct(self):
        space.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 0.5, 2.0, w)  
        
        space.add_concept(42, f)
        self.assertTrue(42 in space._concepts)
        self.assertEqual(space._concepts[42], f)
    
    def test_delete_concept(self):
        space.init(4, {0:[0,1], 1:[2,3]})
        s = Core([Cuboid([1,2,3,4],[3,4,5,6], {0:[0,1], 1:[2,3]})], {0:[0,1], 1:[2,3]})
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        
        f = Concept(s, 0.5, 2.0, w)  
        
        space.add_concept(42, f)
        self.assertTrue(42 in space._concepts)
        self.assertEqual(space._concepts[42], f)
        
        space.delete_concept(43)
        self.assertTrue(42 in space._concepts)
        self.assertEqual(space._concepts[42], f)
        
        space.delete_concept(42)
        self.assertFalse(42 in space._concepts)
        self.assertEqual(len(space._concepts), 0)
        
        space.delete_concept(1337)
        self.assertEqual(len(space._concepts), 0)
        
    # between()
    # 'crisp'
    def test_between_crisp_4D2dom(self):
        space.init(4, {0:[0,1], 1:[2,3]})
        first = [0,0,0,0]
        middle1 = [0.5,1,0.5,0.75]
        middle2 = [0.5,1,1,1.5]
        middle3 = [0.5,0.5,1,1.5]
        middle4 = [0.5,1,1,3.5]
        second = [1,2,2,3]
        self.assertEqual(space.between(first,middle1,second, method="crisp"), 1.0)
        self.assertEqual(space.between(first,middle2,second, method="crisp"), 1.0)
        self.assertEqual(space.between(first,middle3,second, method="crisp"), 0.0)
        self.assertEqual(space.between(first,middle4,second, method="crisp"), 0.0)
        self.assertEqual(space.between(second,middle1,first, method="crisp"), 1.0)
        self.assertEqual(space.between(second,middle2,first, method="crisp"), 1.0)
        self.assertEqual(space.between(second,middle3,first, method="crisp"), 0.0)
        self.assertEqual(space.between(second,middle4,first, method="crisp"), 0.0)

    def test_between_crisp_4D1dom(self):
        space.init(4, {0:[0,1,2,3]})
        first = [0,0,0,0]
        middle1 = [0.5,1,0.5,0.75]
        middle2 = [0.5,1,1,1.5]
        middle3 = [0.5,0.5,1,1.5]
        middle4 = [0.5,1,1,3.5]
        second = [1,2,2,3]
        self.assertEqual(space.between(first,middle1,second, method="crisp"), 0.0)
        self.assertEqual(space.between(first,middle2,second, method="crisp"), 1.0)
        self.assertEqual(space.between(first,middle3,second, method="crisp"), 0.0)
        self.assertEqual(space.between(first,middle4,second, method="crisp"), 0.0)
        self.assertEqual(space.between(second,middle1,first, method="crisp"), 0.0)
        self.assertEqual(space.between(second,middle2,first, method="crisp"), 1.0)
        self.assertEqual(space.between(second,middle3,first, method="crisp"), 0.0)
        self.assertEqual(space.between(second,middle4,first, method="crisp"), 0.0)

    def test_between_crisp_4D4dom(self):
        space.init(4, {0:[0], 1:[1], 2:[2], 3:[3]})
        first = [0,0,0,0]
        middle1 = [0.5,1,0.5,0.75]
        middle2 = [0.5,1,1,1.5]
        middle3 = [0.5,0.5,1,1.5]
        middle4 = [0.5,1,1,3.5]
        second = [1,2,2,3]
        self.assertEqual(space.between(first,middle1,second, method="crisp"), 1.0)
        self.assertEqual(space.between(first,middle2,second, method="crisp"), 1.0)
        self.assertEqual(space.between(first,middle3,second, method="crisp"), 1.0)
        self.assertEqual(space.between(first,middle4,second, method="crisp"), 0.0)
        self.assertEqual(space.between(second,middle1,first, method="crisp"), 1.0)
        self.assertEqual(space.between(second,middle2,first, method="crisp"), 1.0)
        self.assertEqual(space.between(second,middle3,first, method="crisp"), 1.0)
        self.assertEqual(space.between(second,middle4,first, method="crisp"), 0.0)

    # 'soft'
    def test_between_soft_4D2dom(self):
        space.init(4, {0:[0,1], 1:[2,3]})
        first = [0,0,0,0]
        middle1 = [0.5,1,0.5,0.75]
        middle2 = [0.5,1,1,1.5]
        middle3 = [0.5,0.5,1,1.5]
        middle4 = [0.5,1,1,3.5]
        middle5 = [2,3,4,5]
        second = [1,2,2,3]
        dom = {0:2, 1:1}        
        dim = {0:{0:1, 1:1}, 1:{2:3, 3:2.0}}
        w = Weights(dom, dim)
        self.assertAlmostEqual(space.between(first,middle1,second, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle2,second, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle3,second, method="soft"), 0.991147026)
        self.assertAlmostEqual(space.between(first,middle4,second, method="soft"), 0.835214212)
        self.assertAlmostEqual(space.between(first,middle5,second, method="soft"), 0.409900333)
        self.assertAlmostEqual(space.between(second,middle1,first, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle2,first, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle3,first, method="soft"), 0.991147026)
        self.assertAlmostEqual(space.between(second,middle4,first, method="soft"), 0.835214212)
        self.assertAlmostEqual(space.between(second,middle5,first, method="soft"), 0.409900333)
        
        self.assertAlmostEqual(space.between(first,middle1,second, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle2,second, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle3,second, weights=w, method="soft"), 0.9870214460987412)
        self.assertAlmostEqual(space.between(first,middle4,second, weights=w, method="soft"), 0.8845584238298809)
        self.assertAlmostEqual(space.between(first,middle5,second, weights=w, method="soft"), 0.4148810445825626)
        self.assertAlmostEqual(space.between(second,middle1,first, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle2,first, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle3,first, weights=w, method="soft"), 0.9870214460987412)
        self.assertAlmostEqual(space.between(second,middle4,first, weights=w, method="soft"), 0.8845584238298809)
        self.assertAlmostEqual(space.between(second,middle5,first, weights=w, method="soft"), 0.4148810445825626)
        
    def test_between_soft_4D1dom(self):
        space.init(4, {0:[0,1,2,3]})
        first = [0,0,0,0]
        middle1 = [0.5,1,0.5,0.75]
        middle2 = [0.5,1,1,1.5]
        middle3 = [0.5,0.5,1,1.5]
        middle4 = [0.5,1,1,3.5]
        middle5 = [2,3,4,5]
        second = [1,2,2,3]
        dom = {0:1}        
        dim = {0:{0:1, 1:1, 2:3, 3:2.0}}
        w = Weights(dom, dim)
        self.assertAlmostEqual(space.between(first,middle1,second, method="soft"), 0.972566185)
        self.assertAlmostEqual(space.between(first,middle2,second, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle3,second, method="soft"), 0.978828304)
        self.assertAlmostEqual(space.between(first,middle4,second, method="soft"), 0.787274206)
        self.assertAlmostEqual(space.between(first,middle5,second, method="soft"), 0.403647878)
        self.assertAlmostEqual(space.between(second,middle1,first, method="soft"), 0.972566185)
        self.assertAlmostEqual(space.between(second,middle2,first, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle3,first, method="soft"), 0.978828304)
        self.assertAlmostEqual(space.between(second,middle4,first, method="soft"), 0.787274206)
        self.assertAlmostEqual(space.between(second,middle5,first, method="soft"), 0.403647878)

        self.assertAlmostEqual(space.between(first,middle1,second, weights=w, method="soft"), 0.981903451)
        self.assertAlmostEqual(space.between(first,middle2,second, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle3,second, weights=w, method="soft"), 0.987543202)
        self.assertAlmostEqual(space.between(first,middle4,second, weights=w, method="soft"), 0.784485287)
        self.assertAlmostEqual(space.between(first,middle5,second, weights=w, method="soft"), 0.388549374)
        self.assertAlmostEqual(space.between(second,middle1,first, weights=w, method="soft"), 0.981903451)
        self.assertAlmostEqual(space.between(second,middle2,first, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle3,first, weights=w, method="soft"), 0.987543202)
        self.assertAlmostEqual(space.between(second,middle4,first, weights=w, method="soft"), 0.784485287)
        self.assertAlmostEqual(space.between(second,middle5,first, weights=w, method="soft"), 0.388549374)

    def test_between_soft_4D4dom(self):
        space.init(4, {0:[0], 1:[1], 2:[2], 3:[3]})
        first = [0,0,0,0]
        middle1 = [0.5,1,0.5,0.75]
        middle2 = [0.5,1,1,1.5]
        middle3 = [0.5,0.5,1,1.5]
        middle4 = [0.5,1,1,3.5]
        middle5 = [2,3,4,5]
        second = [1,2,2,3]
        dom = {0:2, 1:1, 2:1, 3:3}        
        dim = {0:{0:1}, 1:{1:1}, 2:{2:1}, 3:{3:1}}
        w = Weights(dom, dim)
        self.assertAlmostEqual(space.between(first,middle1,second, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle2,second, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle3,second, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle4,second, method="soft"), 0.8888888888888888)
        self.assertAlmostEqual(space.between(first,middle5,second, method="soft"), 0.4)
        self.assertAlmostEqual(space.between(second,middle1,first, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle2,first, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle3,first, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle4,first, method="soft"), 0.8888888888888888)
        self.assertAlmostEqual(space.between(second,middle5,first, method="soft"), 0.4)
    
        self.assertAlmostEqual(space.between(first,middle1,second, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle2,second, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle3,second, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(first,middle4,second, weights=w, method="soft"), 0.833333333)
        self.assertAlmostEqual(space.between(first,middle5,second, weights=w, method="soft"), 0.405405405)
        self.assertAlmostEqual(space.between(second,middle1,first, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle2,first, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle3,first, weights=w, method="soft"), 1.0)
        self.assertAlmostEqual(space.between(second,middle4,first, weights=w, method="soft"), 0.833333333)
        self.assertAlmostEqual(space.between(second,middle5,first, weights=w, method="soft"), 0.405405405)
    
    # round()
    def test_round(self):
        self.assertEqual(space.round(0.12345678901234567890), 0.1234567890)
        self.assertEqual(space.round(0.987654321987654321), 0.9876543220)
        self.assertEqual(space.round(0.999), 0.999)
    
    # equal()
    def test_equal(self):
        self.assertTrue(space.equal(0.12345678901234567890, 0.1234567890218749576))
        self.assertFalse(space.equal(0.7363248628, 0.7363248629))
        self.assertTrue(space.equal(space.round(0.65348616381658), 0.65348616381658))
        
unittest.main()