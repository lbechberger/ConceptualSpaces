# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 14:39:33 2020

@author: lbechberger
"""
import sys
sys.path.append("..")
import cs.cs
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


def count(n_dims, cuboids_per_concept, num_samples, max_dim_per_domain):

    dimensions = range(n_dims)
    random.seed(42)
    
    greater_than_one = 0
    
    counter = 0
    fails = 0
    while counter < num_samples:
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
        w1 = random_weights(domains)
        w2 = random_weights(domains)
        c1_list = []
        c2_list = []
        for i in range(cuboids_per_concept):
            c1_list.append(random_cuboid(dimensions, domains, 0.0, 1.0))
            c2_list.append(random_cuboid(dimensions, domains, 0.0, 1.0))
        s1 = cs.core.from_cuboids(c1_list, domains)
        s2 = cs.core.from_cuboids(c2_list, domains)
        
        f1 = cs.concept.Concept(s1, random.uniform(0.01, 1.0), random.uniform(1.0, 50.0), w1)
        f2 = cs.concept.Concept(s2, random.uniform(0.01, 1.0), random.uniform(1.0, 50.0), w2)
    
        try:
           if f1.subset_of(f2) > 1:
               greater_than_one += 1
               print(f1,f2)
        except Exception:
            fails += 1
            continue
        
        counter += 1
                
    print("ran {0} examples, failed {1} times, found {2} cases with subsethood > 1".format(counter, fails, greater_than_one))
    return greater_than_one


num_samples = 10000
max_dim_per_domain = 4

for n_dims in [2,4,8]:
    for n_cuboids in [2,4,8]:
        count(n_dims,n_cuboids,num_samples,max_dim_per_domain)
        print(n_dims, n_cuboids)
