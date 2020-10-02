# -*- coding: utf-8 -*-
"""
Test the runtime behavior of our implementation. Requires mpmath

Created on Thu Jun 29 10:07:24 2017

@author: lbechberger
"""
import sys
sys.path.append("..")
import cs.cs
import time
import random
from math import sqrt
import numpy as np
from scipy.integrate import nquad

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
    for dom, dims in domains.items():
        dom_weights[dom] = random.uniform(0.01, 1.0)
        local_dim_weights = {}
        for dim in dims:
            local_dim_weights[dim] = random.uniform(0.01, 1.0)
        dim_weights[dom] = local_dim_weights
    return cs.weights.Weights(dom_weights, dim_weights)        

def runtime(n_dims, cuboids_per_concept, alpha, method, num_samples, max_dim_per_domain, operation):
    """Computes runtime statistics for the givenoperation of concepts.
    
    Parameters: 
        n_dims: number of dimensions
        cuboids_per_concept: number of cuboids per concept
        alpha: number of alpha cuts to use
        method: method of computation to use
        num_samples: number of samples to draw
        max_dim_per_domain: maximal number of dimensions per domain
        operation: operation to time"""

    dimensions = list(range(n_dims))
    random.seed(42)
    
    times = []
    counter = 0
    fails = 0
    while counter < num_samples:
#    for i in range(num_samples):
#        print(i)
        # create a  random domain structure
        domains = {}
        dimensions_left = dimensions
        j = 0
        while len(dimensions_left)  > 0:
            num_dims = random.randint(1, min(len(dimensions_left), max_dim_per_domain))
            dims = random.sample(dimensions_left, num_dims)
            domains[j] = list(dims)
            dimensions_left = [dim for dim in dimensions_left if dim not in dims]
            j += 1

        # make the conceptual space
        cs.cs.init(n_dims, domains)
        
        # create three concepts with random identical weights, random cuboids, maximal mu and random c
        w = random_weights(domains)
        c1_list = []
        c2_list = []
        c3_list = []
        for i in range(cuboids_per_concept):
            c1_list.append(random_cuboid(dimensions, domains, 0.0, 1.0))
            c2_list.append(random_cuboid(dimensions, domains, 0.0, 1.0))
            c3_list.append(random_cuboid(dimensions, domains, 0.0, 1.0))
        s1 = cs.core.from_cuboids(c1_list, domains)
        s2 = cs.core.from_cuboids(c2_list, domains)
        s3 = cs.core.from_cuboids(c3_list, domains)
        
        f1 = cs.concept.Concept(s1, random.uniform(0.01, 1.0), random.uniform(1.0, 50.0), w)
        f2 = cs.concept.Concept(s2, random.uniform(0.01, 1.0), random.uniform(1.0, 50.0), w)
        f3 = cs.concept.Concept(s3, random.uniform(0.01, 1.0), random.uniform(1.0, 50.0), w)

        try:
            start = time.time()
            operation(f1, f2, f3, n_dims, cuboids_per_concept, method, alpha)
            end = time.time()
        except Exception:
            fails += 1
            continue
        duration = end-start
        times.append(duration)
        counter += 1
    min_time = min(times)
    max_time = max(times)
    mean_time = sum(times) / num_samples
    std_time = sqrt(sum([(x - mean_time)**2 for x in times])/num_samples)

    print(("{0},{1},{2},{3},{4},{5},{6},{7},{8}".format(n_dims, cuboids_per_concept, alpha, method, min_time*1000, max_time*1000, mean_time*1000, std_time*1000, fails)))
      
####################################################################################################################################  
# MAIN: here we select what to run at all
config_to_run = 'betweenness'

params = {}
params['intersection'] = [{'n': [1,2,4,8,16,32,64,128,256,512], 'c': [1], 'a': [20], 'm': ['n/a'], 'r': 1000}, 
                          {'n': [8], 'c': [2,4,8,16], 'a': [20], 'm': ['n/a'], 'r': 1000}]
params['size'] = [{'n': [1,2,4,8,16], 'c': [1], 'a': [20], 'm': ['n/a'], 'r': 1000},
                  {'n': [4], 'c': [2,4,8], 'a': [20], 'm': ['n/a'], 'r': 1000}]
params['size_approx'] = [{'n': [1,2], 'c': [1], 'a': [20], 'm': ['n/a'], 'r': 100}]
params['betweenness'] = [{'n': [1,2,4], 'c': [1], 'a': [20], 'm': ['minimum'], 'r': 100},
                         {'n': [1,2,4,8,16], 'c': [1], 'a': [20], 'm': ['integral'], 'r': 100}, 
                         {'n': [2], 'c': [2,4,8], 'a': [20], 'm': ['minimum', 'integral'], 'r': 100},
                         {'n': [2], 'c': [1], 'a': [50,100], 'm': ['minimum', 'integral'], 'r': 100}]

# helper function for the numerical approximation of the size integral
def approx_helper(x, y, z, n, c, m, a):
    borders = [[-np.inf,np.inf]]*n 
    def membership(*args):
        return x.membership_of(args)
    nquad(membership, borders)   

operations = {'intersection': lambda x,y,z,n,c,m,a: x.intersect_with(y),
              'size': lambda x,y,z,n,c,m,a: x.size(),
              'size_approx': lambda x,y,z,n,c,m,a: approx_helper(x,y,z,n,c,m,a),
              'betweenness': lambda x,y,z,n,c,m,a: x.between(y,z, method=m, num_alpha_cuts=a)}

for param_set in params[config_to_run]:
    print(("\n{0} - {1} repetitions".format(config_to_run, param_set['r'])))
    print("n,c,alpha,method,min_time,max_time,mean_time,std_time,fails")
    for n in param_set['n']:
        for c in param_set['c']:
            for a in param_set['a']:
                for m in param_set['m']:
                    runtime(n, c, a, m, param_set['r'], 5, operations[config_to_run])
