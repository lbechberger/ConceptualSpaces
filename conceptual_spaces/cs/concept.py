# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 11:54:30 2017

@author: lbechberger
"""

from math import exp, sqrt, factorial, pi, gamma, log
from random import uniform
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
        
        if (not isinstance(weights, wghts.Weights)) or (not wghts.check(weights._domain_weights, weights._dimension_weights)):
            raise Exception("Invalid weights")
        
        self._core = core
        self._mu = mu
        self._c = c
        self._weights = weights
            
    def __str__(self):
        return "core: {0}\nmu: {1}\nc: {2}\nweights: {3}".format(self._core, self._mu, self._c, self._weights)
    
    def __eq__(self, other):
        if not isinstance(other, Concept):
            return False
        if not (self._core == other._core and cs.equal(self._mu, other._mu) and cs.equal(self._c, other._c) and self._weights == other._weights):
            return False
        return True
    
    def __ne__(self, other):
        return not self.__eq__(other)
    
    def membership_of(self, point):
        """Computes the membership of the point in this concept."""
        
        min_distance = reduce(min, map(lambda x: cs.distance(x, point, self._weights), self._core.find_closest_point_candidates(point)))
        
        return self._mu * exp(-self._c * min_distance)
    
    def _intersection_mu_special_case(self, a, c2, b, mu):
        """Membership of b in c2 (other) to c1 (self) is higher than mu (other)."""

        def makeFun(idx): # need this in order to avoid weird results (defining lambda in loop)
            return (lambda y: y[idx] - b[idx])
        distance = - log(mu / self._mu) / self._c
        y = []
        for i in range(cs._n_dim):
            if a[i] == b[i]:
                y.append(a[i])
            else:
                constr = [{"type":"eq", "fun":(lambda y: cs.distance(a,y,self._weights) - distance)}]
                for j in range(cs._n_dim):
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
        
        crisp_intersection = c1.intersect_with(c2)
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
        if self._mu * exp(-self._c * cs.distance(a, b, self._weights)) >= other._mu:
            # intersection is part of other cuboid
            mu = other._mu
            p_min, p_max = self._intersection_mu_special_case(a, c2, b, mu)
        elif other._mu * exp(-other._c * cs.distance(a, b, other._weights)) >= self._mu:
            # intersection is part of this cuboid
            mu = self._mu
            p_min, p_max = other._intersection_mu_special_case(b, c1, a, mu)
        else:
            # intersection is in the cuboid between a and b
            # --> find point with highest identical membership to both cuboids
        
            # only use the relevant dimensions in order to make optimization easier
            def membership(x, point, mu, c, weights):
                x_new = []
                j = 0
                for dim in range(cs._n_dim):
                    if extrude[dim]:
                        x_new.append(point[dim])
                    else:
                        x_new.append(x[j])
                        j += 1
                return mu * exp(-c * cs.distance(point, x_new, weights))

            bounds = []
            for dim in range(cs._n_dim):
                if not extrude[dim]:
                    bounds.append((min(a[dim], b[dim]), max(a[dim], b[dim])))
            first_guess = map(lambda (x, y): (x + y)/2.0, bounds)
            to_minimize = lambda x: -membership(x, a, self._mu, self._c, self._weights)
            constr = [{"type":"eq", "fun":(lambda x: abs(membership(x, a, self._mu, self._c, self._weights) - membership(x, b, other._mu, other._c, other._weights)))}]
            opt = scipy.optimize.minimize(to_minimize, first_guess, constraints = constr, bounds = bounds, options = {"eps":cs._epsilon}) #, "maxiter":500
            if not opt.success and abs(opt.fun - membership(opt.x, b, other._mu, other._c, other._weights)) < 1e-06:
                # if optimizer failed to find exact solution, but managed to find approximate solution: take it
                raise Exception("Optimizer failed!")
            # reconstruct full x by inserting fixed coordinates that will be extruded later
            x_star = []
            j = 0
            for dim in range(cs._n_dim):
                if extrude[dim]:
                    x_star.append(a[dim])
                else:
                    x_star.append(opt.x[j])
                    j += 1
            mu = membership(opt.x, a, self._mu, self._c, self._weights)

            # check if the weights are linearly dependent w.r.t. all relevant dimensions            
            relevant_dimensions = []
            for i in range(cs._n_dim):
                if not extrude[i]:
                    relevant_dimensions.append(i)
            relevant_domains = self._reduce_domains(cs._domains, relevant_dimensions)
            
            t = None
            weights_dependent = True
            for (dom, dims) in relevant_domains.items():
                for dim in dims:
                    if t is None:
                        # initialize
                        t = (self._weights._domain_weights[dom] * sqrt(self._weights._dimension_weights[dom][dim])) / (other._weights._domain_weights[dom] * sqrt(other._weights._dimension_weights[dom][dim]))
                    else:
                        # compare
                        t_prime = (self._weights._domain_weights[dom] * sqrt(self._weights._dimension_weights[dom][dim])) / (other._weights._domain_weights[dom] * sqrt(other._weights._dimension_weights[dom][dim]))
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
                
                for num_free_dims in range(1, len(relevant_dimensions)):
                    # start with a single free dimensions (i.e., edges of the bounding box) and increase until we find a solution
                    for free_dims in itertools.combinations(relevant_dimensions, num_free_dims):
                        # free_dims is the set of dimensions that are allowed to vary, all other ones are fixed
                        
                        binary_vecs = list(itertools.product([False,True], repeat = len(relevant_dimensions) - num_free_dims))
                        
                        for vec in binary_vecs:
                           
                            # compute the difference between the actual distance and the desired epsilon-distance
                            def epsilon_difference(x, point, weights, epsilon):
                                i = 0
                                j = 0
                                x_new = []
                                # puzzle together our large x vector based on the fixed and the free dimensions
                                for dim in range(cs._n_dim):
                                    if dim in free_dims:
                                        x_new.append(x[i])
                                        i += 1
                                    elif extrude[dim]:
                                        x_new.append(a[dim])
                                    else:
                                        x_new.append(a[dim] if vec[j] else b[dim])
                                        j += 1
                                return abs(cs.distance(point, x_new, weights) - epsilon)
                            
                            bounds = []
                            for dim in free_dims:
                                bounds.append((min(a[dim], b[dim]), max(a[dim], b[dim])))
                            first_guess = map(lambda (x, y): (x + y)/2.0, bounds)
                            to_minimize = lambda x: max(epsilon_difference(x, a, self._weights, epsilon_1)**2, epsilon_difference(x, b, other._weights, epsilon_2)**2)
                            
                            opt = scipy.optimize.minimize(to_minimize, first_guess) #tol = 0.000001
                            if opt.success:
                                dist1 = epsilon_difference(opt.x, a, self._weights, epsilon_1)
                                dist2 = epsilon_difference(opt.x, b, other._weights, epsilon_2)
                                between = True
                                k = 0
                                for dim in free_dims:
                                    if not (min(a[dim], b[dim]) <= opt.x[k] <= max(a[dim], b[dim])):
                                        between = False
                                        break
                                    k += 1
                                # must be between a and b on all free dimensions AND must be a sufficiently good solution
                                if dist1 < 0.00001 and dist2 < 0.00001 and between:
                                    point = []
                                    i = 0
                                    j = 0
                                    # puzzle together our large x vector based on the fixed and the free dimensions
                                    for dim in range(cs._n_dim):
                                        if dim in free_dims:
                                            point.append(opt.x[i])
                                            i += 1
                                        elif extrude[dim]:
                                            point.append(a[dim])
                                        else:
                                            point.append(a[dim] if vec[j] else b[dim])
                                            j += 1
                                    
                                    points.append(point)
                                                        
                    if len(points) > 0:
                        # if we found a solution for num_free_dims: stop looking at higher values for num_free_dims
                        p_min = []
                        p_max = []
                        for i in range(cs._n_dim):
                            p_min.append(max(min(a[i],b[i]), reduce(min, map(lambda x: x[i], points))))
                            p_max.append(min(max(a[i],b[i]), reduce(max, map(lambda x: x[i], points))))
                        break
                
                if p_min == None or p_max == None:  
                    # this should never happen - if the weights are dependent, there MUST be a solution
                    raise Exception("Could not find solution for dependent weights")
                
            else:
                # weights are not linearly dependent: use single-point cuboid
                p_min = list(x_star)
                p_max = list(x_star)
                pass
        
        # round everything, because we only found approximate solutions anyways
        mu = cs.round(mu)
        p_min = map(cs.round, p_min)
        p_max = map(cs.round, p_max)
                                                   
        # extrude in remaining dimensions
        for i in range(len(extrude)):
            if extrude[i]:
                p_max[i] = a_range[i][1]

        # finally, construct a cuboid and return it along with mu
        cuboid = cub.Cuboid(p_min, p_max, new_domains)
        
        return mu, cuboid

    def intersect_with(self, other):
        """Computes the intersection of two concepts."""

        if not isinstance(other, Concept):
            raise Exception("Not a valid concept")

        # intersect all cuboids pair-wise in order to get cuboid candidates
        candidates = []
        for c1 in self._core._cuboids:
            for c2 in other._core._cuboids:
                candidates.append(self._intersect_fuzzy_cuboids(c1, c2, other))
        
        mu = reduce(max, map(lambda x: x[0], candidates))
        cuboids = map(lambda x: x[1], filter(lambda y: cs.equal(y[0],mu), candidates))        
        
        # create a repaired core
        core = cor.from_cuboids(cuboids, cuboids[0]._domains)
        
        # calculate new c and new weights
        c = min(self._c, other._c)
        weights = self._weights.merge_with(other._weights, 0.5, 0.5)
        
        return Concept(core, mu, c, weights)

    def unify_with(self, other):
        """Computes the union of two concepts."""

        if not isinstance(other, Concept):
            raise Exception("Not a valid concept")
        
        core = self._core.unify_with(other._core) 
        mu = max(self._mu, other._mu)
        c = min(self._c, other._c)
        weights = self._weights.merge_with(other._weights, 0.5, 0.5)
        
        return Concept(core, mu, c, weights)
        
    def project_onto(self, domains):
        """Computes the projection of this concept onto a subset of domains."""
        
        # no explicit check for domains - Core will take care of this
        new_core = self._core.project_onto(domains)
        new_weights = self._weights.project_onto(domains)
        
        return Concept(new_core, self._mu, self._c, new_weights)

    def cut_at(self, dimension, value):
        """Computes the result of cutting this concept into two parts (at the given value on the given dimension).
        
        Returns the lower part and the upper part as a tuple (lower, upper)."""
        
        lower_core, upper_core = self._core.cut_at(dimension, value)
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

        all_dims = [dim for domain in self._core._domains.values() for dim in domain]
        n = len(all_dims)

        # calculating the factor in front of the sum
        weight_product = 1.0
        for (dom, dom_weight) in self._weights._domain_weights.items():
            for (dim, dim_weight) in self._weights._dimension_weights[dom].items():
                weight_product *= dom_weight * sqrt(dim_weight)
        factor = self._mu / (self._c**n * weight_product)

        # outer sum
        outer_sum = 0.0        
        for i in range(0, n+1):
            # inner sum
            inner_sum = 0.0
            subsets = list(itertools.combinations(all_dims, i))
            for subset in subsets:
                # first product
                first_product = 1.0
                for dim in set(all_dims) - set(subset):
                    dom = filter(lambda (x,y): dim in y, self._core._domains.items())[0][0]
                    w_dom = self._weights._domain_weights[dom]
                    w_dim = self._weights._dimension_weights[dom][dim]
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

    def size(self):
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
                    intersection = intersection.intersect_with(cuboid)
                inner_sum += self._hypervolume_couboid(intersection)
                
            hypervolume += inner_sum * (-1.0)**(l+1)
        
        return hypervolume

    def subset_of(self, other):
        """Computes the degree of subsethood between this concept and a given other concept."""

        common_domains = {}
        for dom, dims in self._core._domains.iteritems():
            if dom in other._core._domains and other._core._domains[dom] == dims:
                common_domains[dom] = dims
        projected_self = self.project_onto(common_domains)
        projected_other = other.project_onto(common_domains)
        
        intersection = projected_self.intersect_with(projected_other)
        intersection._c = projected_other._c
        intersection._weights = projected_other._weights
        projected_self._c = projected_other._c
        projected_self._weights = projected_other._weights
        subsethood = intersection.size() / projected_self.size()
        return subsethood

    def crisp_subset_of(self, other):
        """Checks whether this concept is a crisp subset of the given other concept."""

        # self._mu must not be greater than other._mu
        if self._mu > other._mu:
            return False

        # core of self must be subset of other's alpha-cut with alpha = self._mu
        corner_points = []
        self_dims = [dim for dims in self._core._domains.values() for dim in dims]
        
        for cuboid in self._core._cuboids:
            binary_vecs = itertools.product([False, True], repeat = len(self_dims))
            for vec in binary_vecs:
                point = []
                j = 0
                for i in range(cs._n_dim):
                    if i in self_dims:
                        point.append(cuboid._p_max[i] if vec[j] else cuboid._p_min[i])
                        j += 1
                    else:
                        point.append(0.0)
                corner_points.append(point)
        
        for point in corner_points:
            if other.membership_of(point) < self._mu:
                return False
        
        # domains on which other is defined must be subset of domains on which self is defined
        for dom, dims in other._core._domains.iteritems():
            if not (dom in self._core._domains and self._core._domains[dom] == dims):
                return False

        # for all dimensions: c * w_dom * sqrt(dim) must not be larger for other than for self
        for dom, dims in other._core._domains.iteritems():
            for dim in dims:
                other_value = other._c * other._weights._domain_weights[dom] * sqrt(other._weights._dimension_weights[dom][dim])
                self_value = self._c * self._weights._domain_weights[dom] * sqrt(self._weights._dimension_weights[dom][dim])
                if other_value > self_value:
                    return False
        
        return True

    def implies(self, other):
        """Computes the degree of implication between this concept and a given other concept."""
        
        return self.subset_of(other)
    
    def similarity_to(self, other, method="Jaccard"):
        """Computes the similarity of this concept to the given other concept.
        
        The following methods are avaliable:
            'Jaccard':                Jaccard similarity index (size of intersection over size of union) - used as default
            'subset':                 degree of subsethood as computed in subset_of()
        """
        
        # project both concepts onto their common domains to find a common ground                              
        common_domains = {}
        for dom, dims in self._core._domains.iteritems():
            if dom in other._core._domains and other._core._domains[dom] == dims:
                common_domains[dom] = dims
        if len(common_domains) == 0:
            # can't really compare them because they have no common domains --> return 0.0
            return 0.0
        projected_self = self.project_onto(common_domains)
        projected_other = other.project_onto(common_domains)

        if method == "Jaccard":
            intersection = projected_self.intersect_with(projected_other)
            union = projected_self.unify_with(projected_other)
            sim = intersection.size() / union.size()
            return sim
        
        elif method == "subset":
            return projected_self.subset_of(projected_other)
            
        else:
            raise Exception("Unknown method")

    def between(self, first, second, method="integral", num_alpha_cuts = 20):
        """Computes the degree to which this concept is between the other two given concepts.
        
        The following methods are avaliable:
            'minimum':  minimum over all alpha-cuts
            'integral': coarse approximation of the integral over all alpha-cuts
        """
        
        # if the three concepts are not defined on the exact same set of domains, we return zero
        if len(self._core._domains.keys()) != len(first._core._domains.keys()):
            return 0.0
        if len(self._core._domains.keys()) != len(second._core._domains.keys()):
            return 0.0
        # now we know that the number of domains is the same --> check whether the domains themselves are the same
        for dom, dims in self._core._domains.iteritems():
            if not (dom in first._core._domains and first._core._domains[dom] == dims):
                return 0.0
            if not (dom in second._core._domains and second._core._domains[dom] == dims):
                return 0.0

        if method == "minimum":
            
            # if self._mu is greater than any of first and second, the result is automatically zero
            if self._mu > first._mu or self._mu > second._mu:
                return 0.0
                
            # if self is a crisp subset of either of first or second, the result is automatically one
            if self.crisp_subset_of(first) or self.crisp_subset_of(second):
                return 1.0

            # for all dimensions: c * w_dom * sqrt(dim) must not be larger for first and second than for self
            for dom, dims in self._core._domains.iteritems():
                for dim in dims:
                    first_value = first._c * first._weights._domain_weights[dom] * sqrt(first._weights._dimension_weights[dom][dim])
                    self_value = self._c * self._weights._domain_weights[dom] * sqrt(self._weights._dimension_weights[dom][dim])
                    second_value = second._c * second._weights._domain_weights[dom] * sqrt(second._weights._dimension_weights[dom][dim])
                    if first_value > self_value and second_value > self_value:
                        return 0.0
            
            first_point = first._core.midpoint()
            second_point = second._core.midpoint()            
            
            # start at each corner of each cuboid to get a good estimation of minimum over all points in self
            corners_min = [c._p_min for c in self._core._cuboids] 
            corners_max = [c._p_max for c in self._core._cuboids]
            
            candidates = [(point, 'min') for point in corners_min] + [(point, 'max') for point in corners_max]                        
            
            candidate_results = []
            tolerance = 0.005   # tolerance with respect to constraint violation, needed to ensure convergence
            for candidate in candidates:
                
                # push the points a bit over the edge to ensure we have some sort of gradient in the beginning
                if candidate[1] == 'min':
                    cand = list(map(lambda x: x - cs._epsilon, candidate[0]))
                else:
                    cand = list(map(lambda x: x + cs._epsilon, candidate[0]))
                
                # start with three different values of alpha to get a good estimate over the minmum over all alphas
                alpha_candidates = [0.05 * self._mu, 0.5 * self._mu, 0.95 * self._mu]
                
                for alpha in alpha_candidates:
                    
                    # inner optimization: point in first and point in second (maximizing over both)                     
                    inner_x = first_point + second_point
                    
                    # function to minimize in inner optimization
                    def neg_betweenness(x_inner,x_outer):
                        x = x_inner[:cs._n_dim]                
                        y = x_outer[:-1]
                        z = x_inner[cs._n_dim:]
                        
                        return -1.0 * cs.between(x, y, z, self._weights, method='soft')
                    
                    def inner_optimization(y):
                        alpha = y[-1]
                        
                        inner_constraints = [{'type':'ineq', 'fun': lambda x: first.membership_of(x[:cs._n_dim]) - alpha - tolerance}, # x in alpha-cut of first
                                             {'type':'ineq', 'fun': lambda x: second.membership_of(x[cs._n_dim:]) - alpha - tolerance}]  # z in alpha-cut of second
                        opt = scipy.optimize.minimize(neg_betweenness, inner_x, args=(y,), method='COBYLA', constraints=inner_constraints, options={'catol':2*tolerance, 'tol':cs._epsilon, 'maxiter':1000, 'rhobeg':0.01})
                        if not opt.success and opt.status != 2: # opt.status = 2 means that we reached the iteration limit
                            print opt
                            raise Exception("optimization failed: {0}".format(opt.message))
                        return opt
                
                    # outer optimization: point in self and alpha (minimizing over both)
                    outer_x = cand + [alpha]
                    outer_constraints = ({'type':'ineq', 'fun': lambda x: self._mu - x[-1]},                                # alpha < self._mu
                                         {'type':'ineq', 'fun': lambda x: x[-1]},                                           # alpha > 0
                                         {'type':'ineq', 'fun': lambda x: self.membership_of(x[:-1]) - x[-1] - tolerance})  # y in alpha-cut of self
                    to_minimize_y = lambda y: -1 * inner_optimization(y).fun
                    opt = scipy.optimize.minimize(to_minimize_y, outer_x, method='COBYLA', constraints=outer_constraints, options={'catol':2*tolerance, 'tol':cs._epsilon, 'maxiter':1000, 'rhobeg':0.01})
                    if not opt.success and opt.status != 2: # opt.status = 2 means that we reached the iteration limit
                        print opt
                        raise Exception("optimization failed: {0}".format(opt.message))
                    candidate_results.append(opt.fun)
            
            return min(candidate_results)
            

        elif method == "integral":
           
            # if self is a crisp subset of either of first or second, the result is automatically one
            if self.crisp_subset_of(first) or self.crisp_subset_of(second):
                return 1.0

            # create list of alpha cuts that we want to compute
            step_size = 1.0 / num_alpha_cuts
            alphas = [step_size*i for i in range(1,num_alpha_cuts+1)]
            intermediate_results = []
            
            for alpha in alphas:

                if alpha > self._mu:                    # alpha-cut of self is empty --> define as 1.0
                    intermediate_results.append(1.0)
                    continue
                
                if alpha > first._mu or alpha > second._mu: # alpha-cut of self is not empty, but one of the others is empty
                    intermediate_results.append(0.0)        # --> define as 0.0
                    continue

                # start with all corner points of all cuboids to get a good estimate of min
                corners_min = [c._p_min for c in self._core._cuboids] 
                corners_max = [c._p_max for c in self._core._cuboids]
                
                # compute the maximal allowable difference to the core wrt each dimension
                difference = [0]*cs._n_dim
                for dom, dims in self._core._domains.iteritems():
                    for dim in dims:
                        difference[dim] = (-1.0 / (self._c * self._weights._domain_weights[dom] * sqrt(self._weights._dimension_weights[dom][dim]))) * log(alpha / self._mu)

                # walk away from each corner as much as possible to get candidate points
                candidates = []                
                for corner in corners_min:
                    candidates.append(map(lambda x, y: x - y, corner, difference))
                for corner in corners_max:
                    candidates.append(map(lambda x, y: x + y, corner, difference))
                
                betweenness_values = []
                for candidate in candidates:
                    
                    # find closest point in alpha-cut to given candidate point
                    to_optimize = lambda x: (alpha - self.membership_of(x))**2
                    opt = scipy.optimize.minimize(to_optimize, candidate, method='Nelder-Mead')
                    if not opt.success:
                        print opt
                        raise Exception("optimization failed: {0}".format(opt.message))
                    
                    self_point = opt.x
                    
                    # compute maximal betweenness for any points x,z in alpha-cut of first and third
                    x_start = first._core.midpoint() + second._core.midpoint()
                    tolerance = 0.002
                    constr = [{'type':'ineq', 'fun': lambda x: first.membership_of(x[:cs._n_dim]) - alpha - tolerance},   # x in alpha-cut of first
                              {'type':'ineq', 'fun': lambda x: second.membership_of(x[cs._n_dim:]) - alpha - tolerance}]  # z in alpha-cut of second
                    def neg_betweenness(x):
                        return -1.0 * cs.between(x[:cs._n_dim], self_point, x[cs._n_dim:], self._weights, method='soft')
                    opt = scipy.optimize.minimize(neg_betweenness, x_start, constraints=constr, method='COBYLA', options={'catol':2*tolerance, 'maxiter':10000, 'rhobeg':0.01})
                    if not opt.success and not opt.status == 2: # opt.status = 2 means that we reached the iteration limit
                        print opt
                        raise Exception("optimization failed: {0}".format(opt.message))
                    betweenness_values.append(-opt.fun)
                
                # minimum over all candidate points in alpha-cut of self
                intermediate_results.append(min(betweenness_values))

            # compute average of alpha-cuts to approximate the overall integral
            return sum(intermediate_results) / len(alphas)

        else:
            raise Exception("Unknown method")

    def sample(self, num_samples):
        """Samples 'num_samples' instances from the concept, based on its membership function."""
        
        # get probability densitiy function by dividing the membership function by the concept's size
        # this ensures that the integral over the function is equal to one.
        size = self.size()
        pdf = lambda x: self.membership_of(x) / size
        
        samples = []
        
        # compute the boundaries to sample from:
        # for each dimension, compute the intersection of membership(x) with y = 0.001
        boundaries = []
        for dim in range(cs._n_dim):
            core_min = float("inf")
            core_max = float("-inf")
            for c in self._core._cuboids:
                core_min = min(core_min, c._p_min[dim])
                core_max = max(core_max, c._p_max[dim])
            
            if core_min == float("-inf") and core_max == float("inf"):
                # concept not defined in this dimension --> use arbitrary interval [-2,+2]
                # TODO: come up with something better
                boundaries.append([-2, 2])
            else:
                # concept defined in this dimensions --> use borders of 0.001-cut
                dom = filter(lambda (x,y): dim in y, self._core._domains.items())[0][0]
                difference = - log(0.001/self._mu) / (self._c * self._weights._domain_weights[dom] * sqrt(self._weights._dimension_weights[dom][dim]))
                boundaries.append([core_min - difference, core_max + difference])
        
        # use rejection sampling to generate the expected number of samples
        while len(samples) < num_samples:
            
            # create a uniform sample based on the boundaries
            candidate = [i for i in range(cs._n_dim)]
            candidate = map(lambda x: uniform(boundaries[x][0], boundaries[x][1]), candidate)
            
            u = uniform(0,1)
            
            if u * (1.1/size) <= pdf(candidate):
                samples.append(candidate)
        
        return samples


def _check_crisp_betweenness(points, first, second):
    """Returns a list of boolean flags indicating which of the given points are strictly between the first and the second concept."""
    
    # store whether the ith point has already be shown to be between the two other cores
    betweenness = [False]*len(points)

    for c1 in first._core._cuboids:
        for c2 in second._core._cuboids:
            
            if not c1._compatible(c2):
                raise Exception("Incompatible cuboids")
            p_min = map(min, c1._p_min, c2._p_min)
            p_max = map(max, c1._p_max, c2._p_max)
            dom_union = dict(c1._domains)
            dom_union.update(c2._domains)         
            bounding_box = cub.Cuboid(p_min, p_max, dom_union)

            local_betweenness = [True]*len(points)                    
            # check if each point is contained in the bounding box
            for i in range(len(points)):
                local_betweenness[i] = bounding_box.contains(points[i])
            
            if reduce(lambda x,y: x or y, local_betweenness) == False:  # no need to check inequalities
                continue
            
            # check additional contraints for each domain
            for domain in dom_union.values():
                if len(domain) < 2: # we can safely ignore one-dimensional domains
                    continue
                
                for i in range(len(domain)):
                    for j in range(i+1, len(domain)):
                        # look at all pairs of dimensions within this domain                                
                        d1 = domain[i]
                        d2 = domain[j]

                        # create list of inequalities
                        inequalities = []
                        def makeInequality(p1, p2, below):
                            sign = -1 if below else 1
                            a = (p2[1] - p1[1]) if p2[0] > p1[0] else (p1[1] - p2[1])
                            b = -abs(p1[0] - p2[0])
                            c = -1 * (a * p1[0] + b * p1[1])
                            return (lambda x: (sign * (a * x[0] + b * x[1] + c) <= 0))

                        # different cases
                        if c2._p_max[d1] > c1._p_max[d1] and c2._p_min[d2] > c1._p_min[d2]:
                            inequalities.append(makeInequality([c1._p_max[d1], c1._p_min[d2]], [c2._p_max[d1], c2._p_min[d2]], False))
                        if c2._p_max[d1] > c1._p_max[d1] and c1._p_max[d2] > c2._p_max[d2]:
                            inequalities.append(makeInequality(c1._p_max, c2._p_max, True))
                        if c2._p_min[d1] > c1._p_min[d1] and c2._p_max[d2] > c1._p_max[d2]:
                            inequalities.append(makeInequality([c1._p_min[d1], c1._p_max[d2]], [c2._p_min[d1], c2._p_max[d2]], True))
                        if c2._p_min[d1] > c1._p_min[d1] and c2._p_min[d2] < c1._p_min[d2]:
                            inequalities.append(makeInequality(c1._p_min, c2._p_min, False))
                        
                        if c1._p_max[d1] > c2._p_max[d1] and c1._p_min[d2] > c2._p_min[d2]:
                            inequalities.append(makeInequality([c1._p_max[d1], c1._p_min[d2]], [c2._p_max[d1], c2._p_min[d2]], False))
                        if c1._p_max[d1] > c2._p_max[d1] and c2._p_max[d2] > c1._p_max[d2]:
                            inequalities.append(makeInequality(c1._p_max, c2._p_max, True))
                        if c1._p_min[d1] > c2._p_min[d1] and c1._p_max[d2] > c2._p_max[d2]:
                            inequalities.append(makeInequality([c1._p_min[d1], c1._p_max[d2]], [c2._p_min[d1], c2._p_max[d2]], True))
                        if c1._p_min[d1] > c2._p_min[d1] and c1._p_min[d2] < c2._p_min[d2]:
                            inequalities.append(makeInequality(c1._p_min, c2._p_min, False))
                        
                        
                        for k in range(len(points)):
                            for ineq in inequalities:
                                local_betweenness[k] = local_betweenness[k] and ineq([points[k][d1], points[k][d2]])

                        if not reduce(lambda x, y: x or y, local_betweenness):
                            break
                    if not reduce(lambda x, y: x or y, local_betweenness):
                            break
                if not reduce(lambda x, y: x or y, local_betweenness):
                            break
                        
            betweenness = map(lambda x, y: x or y, betweenness, local_betweenness)
            if reduce(lambda x, y: x and y, betweenness):
                return betweenness
    
    return betweenness
