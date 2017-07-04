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
from math import sqrt

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
    """Computes runtime statistics for the intersection operation of concepts.
    
    Parameters: n = number of dimensions, num_samples: number of samples to draw, max_dim_per_domain: maximal number of dimensions per domain."""
    
    dimensions = range(n)
    random.seed(1)
    
    times = []
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
        times.append(duration)
    
    min_time = min(times)
    max_time = max(times)
    mean_time = sum(times) / num_samples
    std_time = sqrt(sum(map(lambda x: (x - mean_time)**2, times))/num_samples)

    print "{0},{1},{2},{3},{4}".format(n, min_time*1000, max_time*1000, mean_time*1000, std_time*1000)    

def runtime_hypervolume(n, num_samples, max_dim_per_domain):
    """Computes runtime statistics for the hypervolume operation of concepts.
    
    Parameters: n = number of dimensions, num_samples: number of samples to draw, max_dim_per_domain: maximal number of dimensions per domain."""
    dimensions = range(n)
    random.seed(1)
    
    times = []
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
        
        # create a concept with random weights, random cuboids, random mu and random c
        f = cs.concept.Concept(cs.core.Core([random_cuboid(dimensions, domains, 0.0, 1.0)], domains), random.uniform(0.01, 1.0), random.uniform(1.0, 50.0), random_weights(domains))

        start = timer()
        f.hypervolume()
        end = timer()
        duration = end-start
        times.append(duration)
    
    min_time = min(times)
    max_time = max(times)
    mean_time = sum(times) / num_samples
    std_time = sqrt(sum(map(lambda x: (x - mean_time)**2, times))/num_samples)

    print "{0},{1},{2},{3},{4}".format(n, min_time*1000, max_time*1000, mean_time*1000, std_time*1000)    
  
####################################################################################################################################  
# MAIN: here we select what to run at all
run_intersection = False
run_hypervolume = False
list_of_n = [1,2,4,8,16,32,64,128,256,512]

if run_intersection: 
    print "INTERSECTION"
    print "n, min_time, max_time, mean_time, std_time"
    for n in list_of_n:
        runtime_intersection(n, 10000, 5)

if run_hypervolume: 
    print "HYPERVOLUME"
    print "n, min_time, max_time, mean_time, std_time"
    for n in list_of_n:
        runtime_hypervolume(n, 10000, 5)