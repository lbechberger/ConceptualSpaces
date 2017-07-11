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

# define Granny Smith concept
c_granny_smith = Cuboid([0.55, 0.70, 0.35], [0.6, 0.8, 0.45], domains)
s_granny_smith = Core([c_granny_smith], domains)
w_granny_smith = Weights({"color":1.0, "shape":1.0, "taste":1.0}, w_dim)
granny_smith = Concept(s_granny_smith, 1.0, 25.0, w_granny_smith)
space.add_concept("Granny Smith", granny_smith)

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

def demo():
    """Runs a short demo tour on how to use the implementation."""
    
    def wait_for_user():
        print("\nPress Enter to continue...")
        raw_input()
        print("----------------------------\n")
    
    print("\nWelcome to the demo tour.\n")
    print("It will give you a quick overview of the operations provided by this implementation of the conceptual spaces framework.")
    wait_for_user()
    
    print("A conceptual space can be defined as follows:")
    print('  (1) import cs.cs as space')
    print('  (2) domains = {"color":[0], "shape":[1], "taste":[2]}')
    print('  (3) space.init(3, domains)')
    print("")
    print("The first line imports the module responsible for representing the overall conceptual space.")
    print("")
    print("The second line provides the domain structure of the space.")
    print("In this case, we have three domains: color, shape, and taste.")
    print("Each of them consists of a single dimension.")
    print("The domain structure is thus a dictionary mapping from domain names to list of dimension indices.")
    print("Note that each dimension of the space must belong to exactly one domain.")
    print("")
    print("The third line initializes the space with the desired number of dimensions and the specified domain structure.")
    print("Note that the number of dimension given here and the number of dimensions in the domain structure must match.")
    wait_for_user()
    
    print("We can now define a concept as follows:")
    print('  (1) c_pear = Cuboid([0.5, 0.4, 0.35], [0.7, 0.6, 0.45], domains)')
    print('  (2) s_pear = Core([c_pear], domains)')
    print('  (3) w_dim = {"color":{0:1}, "shape":{1:1}, "taste":{2:1}}')
    print('  (4) w_pear = Weights({"color":0.50, "shape":1.25, "taste":1.25}, w_dim)')
    print('  (5) pear = Concept(s_pear, 1.0, 12.0, w_pear)')
    print("")
    print("The first line defines a cuboid with the support points p_min = [0.5, 0.4, 0.35] and p_max = [0.7, 0.6, 0.45].")
    print("Note that this cuboid is defined on the whole space, as there are values for all three dimensions.")
    print("This is also the reason why we pass the overall domain structure as a second argument - the cuboid is defined on all domains.")
    print("")
    print("The second line builds a core out of this single cuboid.")
    print("In theory, we can give a list of multiple cuboids as a first parameter to this constructor. The only constraint is that these cuboids need to have a nonempty intersection.")
    print("We also need again to specify the set of domains on which this core is defined (which in this case is again the whole space).")
    print("")
    print("The third line defines a set of weights for the dimensions.")
    print("As the sum of dimension weights within each dimension must equal 1.0, and as each domain only contains a single dimension, all the dimension weights are set to 1.0 here.")
    print("")
    print("The fourth line defines the domain weights and the overall weights parameter.")
    print("As one can see, the 'shape' and the 'taste' domain are weighted higher than the 'color' domain in this case.")
    print("Note that the sum of the domain weights must equal the number of domains. If the provided numbers don't add up, the constructor of the Weights class will normalize them automatically.")
    print("")
    print("Finally, the fifth line creates the 'pear' concept.")
    print("We use the core defined in line 2 and the weights defined in line 4.")
    print("The maximal membership is set to 1.0 and the sensitivity parameter (which controls the rate of the membership function's exponential decay) is set to 12.")
    wait_for_user()
    
    print("For convenience, the conceptual space also contains a dictionary for storing concepts.")
    print("We can add our newly created concept to this dictionary under the identifier 'pear' as follows:")
    print('    space.add_concept("pear", pear)')
    print("")
    print("In this file, we have already defined several concepts for different types of fruit along with some properties.")
    print("Variables for fruit concepts (with identifiers in parentheses): pear ('pear'), orange ('orange'), lemon ('lemon'), granny_smith ('Granny Smith'), apple ('apple'), banana ('banana')")
    print("Variables for properties: red ('red'), green, ('green'), blue ('blue'), non_sweet ('nonSweet')")
    print("")
    print("The folder TODO contains some 2D and 3D visualizations of these concepts.")
    wait_for_user()
    
    print("We can display a concept by simply printing it:")
    print("    print(pear")
    print("")
    print("This results in the following output:")
    print('    core: {[0.5, 0.4, 0.35]-[0.7, 0.6, 0.45]}')
    print('    mu: 1.0')
    print('    c: 12.0')
    print("    weights: <{'color': 0.5, 'taste': 1.25, 'shape': 1.25},{'color': {0: 1.0}, 'taste': {2: 1.0}, 'shape': {1: 1.0}}>")
    wait_for_user()
    
    print("We can execute the following operations on a concept c:")
    print("    c.membership_of(x): computes the membership of a point x to the concept c.")
    print("    c.intersect_with(d): computes the intersection of the concepts c and d.")
    print("    c.unify_with(d): computes the unification of the two concepts c and d.")
    print("    c.project_onto(domains): projects the concept c onto the given domains.")
    print("    c.cut_at(dimension, value): cuts the concept c into two parts. The cut is placed at the given value on the given dimension.")
    print("    c.size(): computes the size of the concept c.")
    print("    c.subset_of(d): computes the degree to which the concept c is a subset of the concept d.")
    print("    c.implies(d): computes the degree to which the concept c implies the concept d.")
    print("    c.similarity_to(d): computes the degree of similarity between the concept c and the concept d.")
    print("    c.between(d, e): decides whether the concept c is between the concepts d and e.")
    wait_for_user()
    
    print("Let us illustrate these operations:")
    print("    pear.membership_of([0.6, 0.5, 0.4])")
    print("        1.0")
    print("    pear.membership_of([0.3, 0.2, 0.1])")
    print("        0.0003526621646282561")
    print("    print(pear.intersect_with(apple)")
    print("        core: {[0.5, 0.625, 0.35]-[0.7, 0.625, 0.45]}")
    print("        mu: 0.6872892788")
    print("        c: 10.0")
    print("        weights: <{'color': 0.5, 'taste': 1.125, 'shape': 1.375},{'color': {0: 1.0}, 'taste': {2: 1.0}, 'shape': {1: 1.0}}>")
    wait_for_user()
    
    print("    print(pear.unify_with(apple)")
    print("        core: {[0.5, 0.4, 0.35]-[0.7125, 0.6687500000000001, 0.45625000000000004], [0.5, 0.65, 0.35]-[0.8, 0.8, 0.5], [0.65, 0.65, 0.4]-[0.85, 0.8, 0.55], [0.7, 0.65, 0.45]-[1.0, 0.8, 0.6]}")
    print("        mu: 1.0")
    print("        c: 10.0")
    print("        weights: <{'color': 0.5, 'taste': 1.125, 'shape': 1.375},{'color': {0: 1.0}, 'taste': {2: 1.0}, 'shape': {1: 1.0}}>")
    print("    print(pear.project_onto({'color':[0]})")
    print("        core: {[0.5, -inf, -inf]-[0.7, inf, inf]}")
    print("        mu: 1.0")
    print("        c: 12.0")
    print("        weights: <{'color': 1.0},{'color': {0: 1.0}}>")
    wait_for_user()
    
    print("    first, second = pear.cut_at(1, 0.5)")
    print("    print(first")
    print("        core: {[0.5, 0.4, 0.35]-[0.7, 0.5, 0.45]}")
    print("        mu: 1.0")
    print("        c: 12.0")
    print("        weights: <{'color': 0.5, 'taste': 1.25, 'shape': 1.25},{'color': {0: 1.0}, 'taste': {2: 1.0}, 'shape': {1: 1.0}}>")
    print("    print(second")
    print("        core: {[0.5, 0.5, 0.35]-[0.7, 0.6, 0.45]}")
    print("        mu: 1.0")
    print("        c: 12.0")
    print("        weights: <{'color': 0.5, 'taste': 1.25, 'shape': 1.25},{'color': {0: 1.0}, 'taste': {2: 1.0}, 'shape': {1: 1.0}}>")
    wait_for_user()
    
    print("    apple.size()")
    print("        0.10483333333333335")
    print("    lemon.size()")
    print("        0.013500000000000003")
    print("    granny_smith.subset_of(apple)")
    print("        1.0")
    print("    apple.subset_of(granny_smith)")
    print("        0.11709107083287003")
    print("    apple.implies(red)")
    print("        0.3333333333333332")
    print("    lemon.implies(non_sweet)")
    print("        1.0")
    wait_for_user()
    
    print("    apple.similarity_to(pear)")
    print("        0.004516580942612666")
    print("    pear.similarity_to(apple)")
    print("        0.007635094218859955")
    print("    granny_smith.similarity_to(apple)")
    print("        0.1353352832366129")
    print("    apple.between(lemon, orange)")
    print("        1.0")
    print("    banana.between(granny_smith, pear)")
    print("        0.0")
    wait_for_user()
    
    print("This is already the end of our little tour. We hope it gave you an impression of how you can use our framework.")
    print("Feel free to play around with the fruit space a little bit more by typing in your own operations on the given concepts or by defining new ones.")
    wait_for_user()
    