# -*- coding: utf-8 -*-
"""
Implementation of the 3-dimensional fruit space example

Created on Fri Jun  9 10:39:25 2017

@author: lbechberger
"""

import sys
sys.path.append("..")
import cs.cs

# define the conceptual space
domains = {"color":[0], "shape":[1], "taste":[2]}
space = cs.cs.ConceptualSpace(3, domains)

# standard weights for the dimensions within each domain
w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}

# define pear concept
c_pear = cs.cuboid.Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)
s_pear = cs.core.Core([c_pear], domains)
w_pear = cs.weights.Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)
f_pear = cs.concept.Concept(s_pear, 1.0, 10.0, w_pear)
space.add_concept("pear", f_pear)

# define orange concept
c_orange = cs.cuboid.Cuboid([0.8, 0.9, 0.6], [0.9, 1.0, 0.7], domains)
s_orange = cs.core.Core([c_orange], domains)
w_orange = cs.weights.Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
f_orange = cs.concept.Concept(s_orange, 1.0, 15.0, w_orange)
space.add_concept("orange", f_orange)

# define lemon concept
c_lemon = cs.cuboid.Cuboid([0.7, 0.45, 0.0], [0.8, 0.55, 0.1], domains)
s_lemon = cs.core.Core([c_lemon], domains)
w_lemon = cs.weights.Weights({"color":0.5, "shape":0.5, "taste":2.0}, w_dim)
f_lemon = cs.concept.Concept(s_lemon, 1.0, 20.0, w_lemon)
space.add_concept("orange", f_lemon)

# define GrannySmith concept
c_granny_smith = cs.cuboid.Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
s_granny_smith = cs.core.Core([c_granny_smith], domains)
w_granny_smith = cs.weights.Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
f_granny_smith = cs.concept.Concept(s_granny_smith, 1.0, 25.0, w_granny_smith)
space.add_concept("GrannySmith", f_granny_smith)

# define apple concept
c_apple_1 = cs.cuboid.Cuboid([0.5, 0.65, 0.35], [0.8, 0.8, 0.5], domains)
c_apple_2 = cs.cuboid.Cuboid([0.65, 0.65, 0.4], [0.85, 0.8, 0.55], domains)
c_apple_3 = cs.cuboid.Cuboid([0.7, 0.65, 0.45], [1.0, 0.8, 0.6], domains)
s_apple = cs.core.Core([c_apple_1, c_apple_2, c_apple_3], domains)
w_apple = cs.weights.Weights({"color":0.50, "shape":1.50, "taste":1.00}, w_dim)
f_apple = cs.concept.Concept(s_apple, 1.0, 5.0, w_apple)
space.add_concept("apple", f_apple)

# define banana concept
c_banana_1 = cs.cuboid.Cuboid([0.5, 0.1, 0.35], [0.75, 0.30, 0.55], domains)
c_banana_2 = cs.cuboid.Cuboid([0.7, 0.1, 0.5], [0.8, 0.3, 0.7], domains)
c_banana_3 = cs.cuboid.Cuboid([0.75, 0.1, 0.5], [0.85, 0.3, 1.00], domains)
s_banana = cs.core.Core([c_banana_1, c_banana_2, c_banana_3], domains)
w_banana = cs.weights.Weights({"color":0.75, "shape":1.50, "taste":0.75}, w_dim)
f_banana = cs.concept.Concept(s_banana, 1.0, 4.0, w_banana)
space.add_concept("banana", f_banana)

# define nonSweet property
c_non_sweet = cs.cuboid.Cuboid([float("-inf"), float("-inf"), 0.0], [float("inf"), float("inf"), 0.2], {"taste":[2]})
s_non_sweet = cs.core.Core([c_non_sweet], {"taste":[2]})
w_non_sweet = cs.weights.Weights({"taste":1.0}, {"taste":{2:1.0}})
f_non_sweet = cs.concept.Concept(s_non_sweet, 1.0, 7.0, w_non_sweet)
space.add_concept("nonSweet", f_non_sweet)

# define red property
c_red = cs.cuboid.Cuboid([0.9, float("-inf"), float("-inf")], [1.0, float("inf"), float("inf")], {"color":[0]})
s_red = cs.core.Core([c_red], {"color":[0]})
w_red = cs.weights.Weights({"color":1.0}, {"color":{0:1.0}})
f_red = cs.concept.Concept(s_red, 1.0, 20.0, w_red)
space.add_concept("red", f_red)

# define green property
c_green = cs.cuboid.Cuboid([0.45, float("-inf"), float("-inf")], [0.55, float("inf"), float("inf")], {"color":[0]})
s_green = cs.core.Core([c_green], {"color":[0]})
w_green = cs.weights.Weights({"color":1.0}, {"color":{0:1.0}})
f_green = cs.concept.Concept(s_green, 1.0, 10.0, w_green)
space.add_concept("green", f_green)

# define blue property
c_blue = cs.cuboid.Cuboid([0.2, float("-inf"), float("-inf")], [0.3, float("inf"), float("inf")], {"color":[0]})
s_blue = cs.core.Core([c_blue], {"color":[0]})
w_blue = cs.weights.Weights({"color":1.0}, {"color":{0:1.0}})
f_blue = cs.concept.Concept(s_blue, 1.0, 10.0, w_blue)
space.add_concept("blue", f_blue)


