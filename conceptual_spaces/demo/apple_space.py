# -*- coding: utf-8 -*-
"""
Implementation of the apple example space used in my dissertation for comparison with other approaches.
The apple example space consists of two domains: color (one dimension) and taste (two dimensions).

Created on Thu Aug 23 09:11:08 2018

@author: lbechberger
"""

import sys
sys.path.append("..")
import cs.cs as space
from cs.weights import Weights
from cs.cuboid import Cuboid
from cs.core import Core
from cs.concept import Concept
import visualization.concept_inspector as ci

# define the conceptual space
domains = {"color":[0], "taste":[1, 2]}
dimension_names = ["hue", "sour", "sweet"]
space.init(3, domains, dimension_names)

# define red property
c_red = Cuboid([0.7, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
s_red = Core([c_red], {"color":[0]})
w_red = Weights({"color":1.0}, {"color":{0:1.0}})
red = Concept(s_red, 1.0, 40.0, w_red)
space.add_concept("red", red, 'r')

# define yellow property
c_yellow = Cuboid([0.4, float("-inf"), float("-inf")], [0.6, float("inf"), float("inf")], {"color":[0]})
s_yellow = Core([c_yellow], {"color":[0]})
w_yellow = Weights({"color":1.0}, {"color":{0:1.0}})
yellow = Concept(s_yellow, 1.0, 40.0, w_yellow)
space.add_concept("yellow", yellow, 'y')

# define green property
c_green = Cuboid([0.0, float("-inf"), float("-inf")], [0.3, float("inf"), float("inf")], {"color":[0]})
s_green = Core([c_green], {"color":[0]})
w_green = Weights({"color":1.0}, {"color":{0:1.0}})
green = Concept(s_green, 1.0, 40.0, w_green)
space.add_concept("green", green, 'g')

# define sour property
c_sour = Cuboid([float("-inf"), 0.5, 0.0], [float("inf"), 1.0, 0.4], {"taste":[1,2]})
s_sour = Core([c_sour], {"taste":[1,2]})
w_sour = Weights({"taste":1.0}, {"taste":{1:0.7, 2:1.0}})
sour = Concept(s_sour, 1.0, 14.0, w_sour)
space.add_concept("sour", sour, 'gray')

# define sweet property
c_sweet = Cuboid([float("-inf"), 0.0, 0.5], [float("inf"), 0.4, 1.0], {"taste":[1,2]})
s_sweet = Core([c_sweet], {"taste":[1,2]})
w_sweet = Weights({"taste":1.0}, {"taste":{1:0.3, 2:0.7}})
sweet = Concept(s_sweet, 1.0, 14.0, w_sweet)
space.add_concept("sweet", sweet, 'b')

# define apple concept
c_apple_1 = Cuboid([0.1, 0.5, 0.1], [0.55, 0.9, 0.5], domains)
c_apple_2 = Cuboid([0.3, 0.3, 0.4], [0.7, 0.6, 0.55], domains)
c_apple_3 = Cuboid([0.45, 0.1, 0.45], [0.9, 0.5, 0.9], domains)
s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
w_apple = Weights({"color":0.67, "taste":1.33}, {"color":{0:1.0}, "taste":{1:1.0, 2:1.0}})
apple = Concept(s_apple, 1.0, 20.0, w_apple)
space.add_concept("apple", apple, 'r')

ci.init()