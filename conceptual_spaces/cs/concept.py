# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: lbechberger
"""

from math import exp, sqrt, factorial, pi, gamma, log
import itertools
import scipy.optimize

import core as cor
import cuboid as cub
import weights as wghts
import cs

class Concept:
    """A concept, implementation of the Fuzzy Simple Star-Shaped Set (FSSSS)."""
    
    def __init__(self, core, mu, c, weights):
        """Initializes the concept."""

        if (not isinstance(core, cor.Core)) or (not cor.check(core._cuboids, core._domains)):
            raise Exception("Invalid core")

        if mu > 1.0 or mu <= 0.0:
            raise Exception("Invalid mu")
        
        if c <= 0.0:
            raise Exception("Invalid c")
        
        if (not isinstance(weights, wghts.Weights)) or (not weights._check()):
            raise Exception("Invalid weights")
        
        self._core = core
        self._mu = mu
        self._c = c
        self._weights = weights
            
    def __str__(self):
        return "<{0},{1},{2},{3}>".format(self._core, self._mu, self._c, self._weights)
    
    def __eq__(self, other):
        if not isinstance(other, Concept):
            return False
        if self._core != other._core or self._mu != other._mu or self._c != other._c or self._weights != other._weights:
            return False
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def membership(self, point):
        """Computes the membership of the point in this concept."""
        
        min_distance = reduce(min, map(lambda x: cs.ConceptualSpace.cs.distance(x, point, self._weights), self._core.find_closest_point_candidates(point)))
        
        return self._mu * exp(-self._c * min_distance)
    
    def _intersection_mu_special_case(self, a, c2, b, mu):
        """Membership of b in c2 (other) to c1 (self) is higher than mu (other)."""

        def makeFun(idx): # need this in order to avoid weird results (defining lambda in loop)
            return (lambda y: y[idx] - b[idx])
        distance = - log(mu / self._mu) / self._c
        y = []
        for i in range(cs.ConceptualSpace.cs._n_dim):
            if a[i] == b[i]:
                y.append(a[i])
            else:
                constr = [{"type":"eq", "fun":(lambda y: cs.ConceptualSpace.cs.distance(a,y,self._weights) - distance)}]
                for j in range(cs.ConceptualSpace.cs._n_dim):
                    if i != j:
                        constr.append({"type":"eq", "fun":makeFun(j)})
                
                if a[i] < b[i]:
                    opt = scipy.optimize.minimize(lambda y: -y[i], b, constraints = constr)
                    if not opt.success:
                        raise Exception("Optimizer failed!")
                    y.append(opt.x[i])
                else: 
                    opt = scipy.optimize.minimize(lambda y: y[i], b, constraints = constr)
                    if not opt.success:
                        raise Exception("Optimizer failed!")
                    y.append(opt.x[i])
        
        # arrange entries in b and y to make p_min and p_max; make sure we don't fall out of c2
        p_min = map(max, map(min, b, y), c2._p_min)
        p_max = map(min, map(max, b, y), c2._p_max)
        
        # take the unification of domains
        return p_min, p_max
    
    def _intersect_fuzzy_cuboids(self, c1, c2, other):
        """Find the highest intersection of the two cuboids (c1 from this, c2 from the other concept)."""
        
        crisp_intersection = c1.intersect(c2)
        if (crisp_intersection != None):  # crisp cuboids already intersect
            return min(self._mu, other._mu), crisp_intersection
        
        # already compute new set of domains
        new_domains = dict(c1._domains)
        new_domains.update(c2._domains)         
        
        # get ranges of closest points, store which dimensions need to be extruded, pick example points
        a_range, b_range = c1.get_closest_points(c2)
        a = map(lambda x: x[0], a_range)
        b = map(lambda x: x[0], b_range)
        extrude = map(lambda x: x[0] != x[1], a_range)
        
        mu = None
        p_min = None
        p_max = None
        if self._mu * exp(-self._c * cs.ConceptualSpace.cs.distance(a, b, self._weights)) >= other._mu:
            # intersection is part of other cuboid
            mu = other._mu
            p_min, p_max = self._intersection_mu_special_case(a, c2, b, mu)
        elif other._mu * exp(-other._c * cs.ConceptualSpace.cs.distance(a, b, other._weights)) >= self._mu:
            # intersection is part of this cuboid
            mu = self._mu
            p_min, p_max = other._intersection_mu_special_case(b, c1, a, mu)
        else:
            # intersection is in the cuboid between a and b
        
            # find point with highest identical membership to both cuboids
            neg_self_membership = lambda x: -self._mu * exp(-self._c * cs.ConceptualSpace.cs.distance(a, x, self._weights))
            neg_other_membership = lambda x: -other._mu * exp(-other._c * cs.ConceptualSpace.cs.distance(b, x, other._weights))
            start = map(lambda x, y: (x + y)/2.0, a, b)
            constr = [{"type":"eq", "fun":(lambda x: neg_self_membership(x) - neg_other_membership(x))}]
            def makeFunMin(idx): # need this in order to avoid weird results (defining lambda in loop)
                return (lambda x: x[idx] - min(a[idx],b[idx]))
            def makeFunMax(idx): # need this in order to avoid weird results (defining lambda in loop)
                return (lambda x: max(a[idx],b[idx]) - x[idx])
            for i in range(len(a)):
                constr.append({"type":"ineq", "fun":makeFunMin(i)})#(lambda x: x[i] - min(a[i], b[i]))})
                constr.append({"type":"ineq", "fun":makeFunMax(i)})#(lambda x: max(a[i], b[i]) - x[i])})
            opt = scipy.optimize.minimize(neg_self_membership, start, constraints = constr)
            if not opt.success:
                raise Exception("Optimizer failed!")
            x_star = list(opt.x)
            mu = -neg_self_membership(x_star)

            # check if the weights are linearly dependent w.r.t. all relevant dimensions            
            relevant_dimensions = []
            for i in range(cs.ConceptualSpace.cs._n_dim):
                if not extrude[i]:
                    relevant_dimensions.append(i)
            relevant_domains = self._reduce_domains(cs.ConceptualSpace.cs._domains, relevant_dimensions)
            
            t = None
            weights_dependent = True
            for (dom, dims) in relevant_domains.items():
                for dim in dims:
                    if t is None:
                        # initialize
                        t = (self._weights.domain_weights[dom] * sqrt(self._weights.dimension_weights[dom][dim])) / (other._weights.domain_weights[dom] * sqrt(other._weights.dimension_weights[dom][dim]))
                    else:
                        # compare
                        t_prime = (self._weights.domain_weights[dom] * sqrt(self._weights.dimension_weights[dom][dim])) / (other._weights.domain_weights[dom] * sqrt(other._weights.dimension_weights[dom][dim]))
                        if round(t,10) != round(t_prime,10):
                            weights_dependent = False
                            break
                if not weights_dependent:
                    break
            
            if weights_dependent and len(relevant_domains.keys()) > 1:   
                # weights are linearly dependent and at least two domains are involved
                # --> need to find all possible corner points of resulting cuboid
                epsilon_1 = - log(mu / self._mu) / self._c
                epsilon_2 = - log(mu / other._mu) / other._c
                points = []
                
                to_minimize = lambda x: abs(cs.ConceptualSpace.cs.distance(a, x, self._weights) - epsilon_1)
                constr = [{"type":"eq", "fun":(lambda x: cs.ConceptualSpace.cs.distance(b, x, other._weights) - epsilon_2)}]
                for free_dim in relevant_dimensions:
                    # free_dim is the single dimension that is allowed to vary, all other ones are fixed
                    binary_vecs = list(itertools.product([False,True], repeat = len(relevant_dimensions) - 1))
                    def makeFun(idx, point): # need this in order to avoid weird results (defining lambda in loop)
                        return (lambda x: x[idx] - point[idx])
                    for vec in binary_vecs:
                        local_constr = list(constr)
                        i = 0
                        for dim in relevant_dimensions:
                            if dim == free_dim:
                                continue
                            if vec[i]:
                                local_constr.append({"type":"eq", "fun":makeFun(dim,a)})#(lambda x: x[dim] - a[dim])})
                            else:
                                local_constr.append({"type":"eq", "fun":makeFun(dim,b)})#(lambda x: x[dim] - b[dim])})
                            i += 1
                        first_guess = map(lambda x, y: (x + y)/2.0, a, b)
                        opt = scipy.optimize.minimize(to_minimize, first_guess, constraints = local_constr)
                        if opt.success and min(a[free_dim], b[free_dim]) <= opt.x[free_dim] <= max(a[free_dim], b[free_dim]):
                            # only keep useful points - ignore the ones that are outside of the bounding box
                            points.append(opt.x)
                
                p_min = []
                p_max = []
                for i in range(cs.ConceptualSpace.cs._n_dim):
                    p_min.append(max(min(a[i],b[i]), reduce(min, map(lambda x: x[i], points))))
                    p_max.append(min(max(a[i],b[i]), reduce(max, map(lambda x: x[i], points))))
            
            else:
                # weights are not linearly dependent: use single-point cuboid
                p_min = list(x_star)
                p_max = list(x_star)
                pass
                                                           
        # extrude in remaining dimensions
        for i in range(len(extrude)):
            if extrude[i]:
                p_max[i] = a_range[i][1]

        # round for tests
        mu = round(mu, 10)
        p_min = map(lambda x: round(x, 10), p_min)
        p_max = map(lambda x: round(x, 10), p_max)

        # finally, construct a cuboid and return it along with mu
        cuboid = cub.Cuboid(p_min, p_max, new_domains)
        
        return mu, cuboid

    def intersect(self, other):
        """Computes the intersection of two concepts."""

        if not isinstance(other, Concept):
            raise Exception("Not a valid concept")

        # intersect all cuboids pair-wise in order to get cuboid candidates
        candidates = []
        for c1 in self._core._cuboids:
            for c2 in other._core._cuboids:
                candidates.append(self._intersect_fuzzy_cuboids(c1, c2, other))
        
        mu = reduce(max, map(lambda x: x[0], candidates))
        cuboids = map(lambda x: x[1], filter(lambda y: y[0] == mu, candidates))        
        
        # create a repaired core
        core = cor.from_cuboids(cuboids, cuboids[0]._domains)
        
        # calculate new c and new weights
        c = min(self._c, other._c)
        weights = self._weights.merge(other._weights, 0.5, 0.5)
        
        return Concept(core, mu, c, weights)

    def unify(self, other):
        """Computes the union of two concepts."""

        if not isinstance(other, Concept):
            raise Exception("Not a valid concept")
        
        core = self._core.unify(other._core) 
        mu = max(self._mu, other._mu)
        c = min(self._c, other._c)
        weights = self._weights.merge(other._weights, 0.5, 0.5)
        
        return Concept(core, mu, c, weights)
        
    def project(self, domains):
        """Computes the projection of this concept onto a subset of domains."""
        
        # no explicit check for domains - Core will take care of this
        new_core = self._core.project(domains)
        new_weights = self._weights.project(domains)
        
        return Concept(new_core, self._mu, self._c, new_weights)

    def cut(self, dimension, value):
        """Computes the result of cutting this concept into two parts (at the given value on the given dimension).
        
        Returns the lower part and the upper part as a tuple (lower, upper)."""
        
        lower_core, upper_core = self._core.cut(dimension, value)
        lower_concept = None if lower_core == None else Concept(lower_core, self._mu, self._c, self._weights)
        upper_concept = None if upper_core == None else Concept(upper_core, self._mu, self._c, self._weights)
        
        return lower_concept, upper_concept

    def _reduce_domains(self, domains, dimensions):
        """Reduces the domain structure such that only the given dimensions are still contained."""
        new_domains = {}

        for (dom, dims) in domains.items():
            filtered_dims = [dim for dim in set(dims) & set(dimensions)]
            if len(filtered_dims) > 0:
                new_domains[dom] = filtered_dims
        
        return new_domains

    def _hypervolume_couboid(self, cuboid):
        """Computes the hypervolume of a single fuzzified cuboid."""

        n = cs.ConceptualSpace.cs._n_dim

        # calculating the factor in front of the sum
        weight_product = 1.0
        for (dom, dom_weight) in self._weights.domain_weights.items():
            for (dim, dim_weight) in self._weights.dimension_weights[dom].items():
                weight_product *= dom_weight * sqrt(dim_weight)
        factor = self._mu / (self._c**n * weight_product)

        all_dims = [dim for domain in self._core._domains.values() for dim in domain]
        outer_sum = 0.0        
        # outer sum
        for i in range(0, n+1):
            subsets = list(itertools.combinations(all_dims, i))
            inner_sum = 0.0
            # inner sum
            for subset in subsets:
                # first product
                first_product = 1.0
                for dim in set(all_dims) - set(subset):
                    dom = filter(lambda (x,y): dim in y, self._core._domains.items())[0][0]
                    w_dom = self._weights.domain_weights[dom]
                    w_dim = self._weights.dimension_weights[dom][dim]
                    b = cuboid._p_max[dim] - cuboid._p_min[dim]
                    first_product *= w_dom * sqrt(w_dim) * b * self._c
                
                # second product
                second_product = 1.0
                reduced_domain_structure = self._reduce_domains(self._core._domains, subset)
                for (dom, dims) in reduced_domain_structure.items():
                    n_domain = len(dims)
                    second_product *= factorial(n_domain) * (pi ** (n_domain/2.0))/(gamma((n_domain/2.0) + 1))
                
                inner_sum += first_product * second_product
            
            outer_sum += inner_sum
        return factor * outer_sum

    def hypervolume(self):
        """Computes the hypervolume of this concept."""
        
        hypervolume = 0.0
        num_cuboids = len(self._core._cuboids)
        
        # use the inclusion-exclusion formula over all the cuboids
        for l in range(1, num_cuboids + 1):
            inner_sum = 0.0

            subsets = list(itertools.combinations(self._core._cuboids, l))           
            for subset in subsets:
                intersection = subset[0]
                for cuboid in subset:
                    intersection = intersection.intersect(cuboid)
                inner_sum += self._hypervolume_couboid(intersection)
                
            hypervolume += inner_sum * (-1.0)**(l+1)
        
        return hypervolume

    def subset_of(self, other):
        """Computes the degree of subsethood between this concept and a given other concept."""
        pass #TODO implement

    def implies(self, other):
        """Computes the degree of implication between this concept and a given other concept."""
        pass #TODO implement
    
    def similarity(self, other):
        """Computes the similarity of this concept to the given other concept."""
        pass #TODO implement

    def between(self, first, second):
        """Computes the degree to which this concept is between the other two given concepts."""
        pass #TODO implement