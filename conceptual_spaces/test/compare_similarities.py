# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:26:56 2017

@author: lbechberger
"""

execfile("../demo/fruit_space.py")

fruits = ["apple", "banana", "pear", "lemon", "orange", "Granny Smith"]
colors = ["red", "green", "blue"]
methods = ["naive", "Jaccard", "subset", "min_core", "max_core", "Hausdorff_core", "min_membership_core", 
           "max_membership_core", "min_center", "max_center", "Hausdorff_center", "min_membership_center", "max_membership_center"]

# raw values
print("first, second, {0}".format(", ".join(methods)))
value_dict = {}
for method in methods:
    value_dict[method] = []
for first_fruit in fruits:
    for second_fruit in fruits:
        similarities = []
        f1 = space._concepts[first_fruit]
        f2 = space._concepts[second_fruit]
        for method in methods:
            try:
                sim = f1.similarity_to(f2, method)
                similarities.append(sim)
                value_dict[method].append(sim)
            except Exception:
                similarities.append(float("NaN"))
                value_dict[method].append(float("NaN"))
        print("{0}, {1}, {2}".format(first_fruit, second_fruit, ", ".join(map(lambda x: str(x), similarities))))

print("\n")
# histogramm data
print("lower_bound, upper_bound, {0}".format(", ".join(methods)))
num_bins = 20
bin_size = 1.0/num_bins

for i in range(num_bins + 1):
    counts = []
    lower_bound = i * bin_size
    upper_bound = (i + 1) * bin_size
    for method in methods:
        counts.append(len(filter(lambda x: lower_bound <= x < upper_bound, value_dict[method])))
    print("{0}, {1}, {2}".format(str(lower_bound), str(upper_bound), ", ".join(map(lambda x: str(x), counts))))