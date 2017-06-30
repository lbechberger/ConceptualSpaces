# -*- coding: utf-8 -*-
"""
Test the runtime behavior of our implementation. Requires mpmath

Created on Thu Jun 29 10:07:24 2017

@author: lbechberger
"""
import sys
sys.path.append("..")
import cs.cs
from timeit import default_timer as timer
import random

def random_cuboid(dimensions, domains, min_val, max_val):
    p_min = []
    p_max = []
    for dim in dimensions:
        p_min.append(random.uniform(0, ((min_val+max_val)/2) - 0.01))
        p_max.append(random.uniform(((min_val+max_val)/2) + 0.01, max_val))
    return cs.cuboid.Cuboid(p_min, p_max, domains)

def random_weights(domains):
    dim_weights = {}
    dom_weights = {}
    for dom, dims in domains.iteritems():
        dom_weights[dom] = random.uniform(0.01, 1.0)
        local_dim_weights = {}
        for dim in dims:
            local_dim_weights[dim] = random.uniform(0.01, 1.0)
        dim_weights[dom] = local_dim_weights
    return cs.weights.Weights(dom_weights, dim_weights)        

def runtime_intersection(n, num_samples, max_dim_per_domain):
    """Computes runtime statistics for the intersection operation of concepts. The parameter 'n' gives the dimensionality of the concepts"""
    dimensions = range(n)
    random.seed(1)
    min_time = 99999
    max_time = 0
    avg_time = 0.0    
    
    for i in range(num_samples):
        # create a  random domain structure
        domains = {}
        dimensions_left = dimensions
        j = 0
        while len(dimensions_left)  > 0:
            num_dims = random.randint(1, min(len(dimensions_left), max_dim_per_domain))
            dims = random.sample(dimensions_left, num_dims)
            domains[j] = list(dims)
            dimensions_left = [dim for dim in dimensions_left if dim not in dims]
            j+= 1
           
        # make the conceptual space
        cs.cs.init(n, domains)
        
        # create two concepts with random identical weights, random cuboids, maximal mu and random c
        w = random_weights(domains)
        f1 = cs.concept.Concept(cs.core.Core([random_cuboid(dimensions, domains, 0.0, 1.0)], domains), 1.00, random.uniform(1.0, 50.0), w)
        f2 = cs.concept.Concept(cs.core.Core([random_cuboid(dimensions, domains, 0.0, 1.0)], domains), 1.00, random.uniform(1.0, 50.0), w)

        start = timer()
        f1.intersect(f2)
        end = timer()
        duration = end-start
        min_time = min(min_time, duration)
        max_time = max(max_time, duration)
        avg_time += duration
        
        
    avg_time /= num_samples
    print "Number of dimensions: {0}\t Number of samples: {1}".format(n, num_samples)
    print "min: {0} ms".format(min_time*1000)
    print "max: {0} ms".format(max_time*1000)
    print "avg: {0} ms\n".format(avg_time*1000)
    
# MAIN: here we select what to run at all
for n in [1,2,3,4,5,10,15,20,30,40,50,100,200,500]:
    runtime_intersection(n, 10000, 2)
