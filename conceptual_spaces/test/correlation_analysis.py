# -*- coding: utf-8 -*-
"""
Creates scatter plots comparing the results of different variants for computing conceptual betweenness.

Created on Thu Sep  5 20:35:37 2019

@author: lbechberger
"""

import sys
sys.path.append("..")
import cs.cs
import random
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

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

def scatter(n_dims, cuboids_per_concept, params, num_samples, max_dim_per_domain, operation):
    """Creates scatter plots for the betweenness values returned by different combinations of alphas and methods.
    
    Parameters: 
        n_dims: number of dimensions
        cuboids_per_concept: number of cuboids per concept
        params: a dictionary mapping from configuration names to a dictionary of named parameters for the operation
        num_samples: number of samples to draw
        max_dim_per_domain: maximal number of dimensions per domain
        operation: operation to evaluate"""

    dimensions = list(range(n_dims))
    random.seed(42)
    
    results = {}
    for key, value in params.items():
        results[key] = []
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

        local_res = {}
        try:
            for config_name, param_dict in params.items():
                    local_res[config_name] = operation(f1, f2, f3, param_dict)
        except Exception:
            fails += 1
            continue
        
        for key, res in local_res.items():
            results[key].append(res)
        counter += 1
        
        if counter % 50 == 0:
            print(("{0}/{1} ...".format(counter, fails)))
        
    print(("ran {0} examples, failed {1} times".format(counter, fails)))

    # all pairs of configurations
    for first_config, second_config in combinations(list(results.keys()), 2):
        
        # draw the plot
        fig, ax = plt.subplots(figsize=(12,12))
        ax.tick_params(axis="x", labelsize=16)
        ax.tick_params(axis="y", labelsize=16)
        ax.set_xlim(-0.01,1.01)
        ax.set_ylim(-0.01,1.01)
        ax.scatter(results[first_config], results[second_config])
        plt.xlabel(first_config, fontsize = 20)
        plt.ylabel(second_config, fontsize = 20)
        plt.show()
        
        # compute the correlations
        pearson, _ = pearsonr(results[first_config], results[second_config])
        spearman, _ = spearmanr(results[first_config], results[second_config])
        print(('{0} - {1}: Pearson {2}, Spearman {3}'.format(first_config, second_config, pearson, spearman)))

    
      
####################################################################################################################################  
# MAIN: here we select what to run at all
config_to_run = 'betweenness'

params = {}
params['similarity'] = {r'$Sim_S$': {'method': 'subset'},
                         r'$Sim_J$': {'method': 'Jaccard'}}
params['betweenness'] = {r'$B_{soft}^{min}$': {'method': 'minimum'},
                         r'$B_{soft}^{int}$ (20 $\alpha$-cuts)': {'method': 'integral', 'num_alpha_cuts': 20},
                         r'$B_{soft}^{int}$ (100 $\alpha$-cuts)': {'method': 'integral', 'num_alpha_cuts': 100}}

config = {}
config['similarity'] = {'number_of_samples': 1000, 'number_of_dimensions': 4, 'max_dim_per_dom': 4, 'number_of_cuboids_per_concept': 2}
config['betweenness'] = {'number_of_samples': 1000, 'number_of_dimensions': 4, 'max_dim_per_dom': 4, 'number_of_cuboids_per_concept': 2}


operations = {'similarity': lambda x,y,z,p: x.similarity_to(y,**p),
              'betweenness': lambda x,y,z,p: x.between(y,z,**p)}

print((config_to_run, config[config_to_run]))
scatter(config[config_to_run]['number_of_dimensions'], config[config_to_run]['number_of_cuboids_per_concept'], params[config_to_run], config[config_to_run]['number_of_samples'], config[config_to_run]['max_dim_per_dom'], operations[config_to_run])
