# -*- coding: utf-8 -*-
"""
Implementation of the 3-dimensional fruit space example.

Created on Fri Jun  9 10:39:25 2017

@author: lbechberger
"""

import sys
sys.path.append("..")
import cs.cs as space
from cs.weights import Weights
from cs.cuboid import Cuboid
from cs.core import Core
from cs.concept import Concept

# define the conceptual space
domains = {"color":[0], "shape":[1], "taste":[2]}
space.init(3, domains)

# standard weights for the dimensions within each domain
w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}

# define pear concept
c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
s_pear = Core([c_pear], domains)
w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
pear = Concept(s_pear, 1.0, 12.0, w_pear)
space.add_concept("pear", pear)

# define orange concept
c_orange = Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
s_orange = Core([c_orange], domains)
w_orange = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
orange = Concept(s_orange, 1.0, 15.0, w_orange)
space.add_concept("orange", orange)

# define lemon concept
c_lemon = Cuboid([0.7, 0.45, 0.0], [0.8, 0.55, 0.1], domains)
s_lemon = Core([c_lemon], domains)
w_lemon = Weights({"color":0.5, "shape":0.5, "taste":2.0}, w_dim)
lemon = Concept(s_lemon, 1.0, 20.0, w_lemon)
space.add_concept("lemon", lemon)

# define GrannySmith concept
c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
s_granny_smith = Core([c_granny_smith], domains)
w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
granny_smith = Concept(s_granny_smith, 1.0, 25.0, w_granny_smith)
space.add_concept("GrannySmith", granny_smith)

# define apple concept
c_apple_1 = Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
c_apple_2 = Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
c_apple_3 = Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
s_apple = Core([c_apple_1, c_apple_2, c_apple_3], domains)
w_apple = Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
apple = Concept(s_apple, 1.0, 10.0, w_apple)
space.add_concept("apple", apple)

# define banana concept
c_banana_1 = Cuboid([0.5, 0.1, 0.35], [0.75, 0.30, 0.55], domains)
c_banana_2 = Cuboid([0.7, 0.1, 0.5], [0.8, 0.3, 0.7], domains)
c_banana_3 = Cuboid([0.75, 0.1, 0.5], [0.85, 0.3, 1.00], domains)
s_banana = Core([c_banana_1, c_banana_2, c_banana_3], domains)
w_banana = Weights({"color":0.75, "shape":1.50, "taste":0.75}, w_dim)
banana = Concept(s_banana, 1.0, 10.0, w_banana)
space.add_concept("banana", banana)

# define nonSweet property
c_non_sweet = Cuboid([float("-inf"), float("-inf"), 0.0], [float("inf"), float("inf"), 0.2], {"taste":[2]})
s_non_sweet = Core([c_non_sweet], {"taste":[2]})
w_non_sweet = Weights({"taste":1.0}, {"taste":{2:1.0}})
non_sweet = Concept(s_non_sweet, 1.0, 7.0, w_non_sweet)
space.add_concept("nonSweet", non_sweet)

# define red property
c_red = Cuboid([0.9, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
s_red = Core([c_red], {"color":[0]})
w_red = Weights({"color":1.0}, {"color":{0:1.0}})
red = Concept(s_red, 1.0, 20.0, w_red)
space.add_concept("red", red)

# define green property
c_green = Cuboid([0.45, float("-inf"), float("-inf")], [0.55, float("inf"), float("inf")], {"color":[0]})
s_green = Core([c_green], {"color":[0]})
w_green = Weights({"color":1.0}, {"color":{0:1.0}})
green = Concept(s_green, 1.0, 20.0, w_green)
space.add_concept("green", green)

# define blue property
c_blue = Cuboid([0.2, float("-inf"), float("-inf")], [0.3, float("inf"), float("inf")], {"color":[0]})
s_blue = Core([c_blue], {"color":[0]})
w_blue = Weights({"color":1.0}, {"color":{0:1.0}})
blue = Concept(s_blue, 1.0, 20.0, w_blue)
space.add_concept("blue", blue)
