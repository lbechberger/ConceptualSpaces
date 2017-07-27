# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 15:26:56 2017

@author: lbechberger
"""

execfile("../demo/fruit_space.py")

fruits = ["apple", "banana", "pear", "lemon", "orange", "Granny Smith"]
colors = ["red", "green", "blue"]
methods = ["naive", "naive_soft", "subset", "core", "core_soft", "Derrac_Schockaert"]

# raw values
print("first, second, third, {0}".format(", ".join(methods)))
value_dict = {}
for method in methods:
    value_dict[method] = []
for first_fruit in fruits:
    for second_fruit in fruits:
        for third_fruit in fruits:
            betweennesses = []
            f1 = space._concepts[first_fruit]
            f2 = space._concepts[second_fruit]
            f3 = space._concepts[third_fruit]
            for method in methods:
                try:
                    btw = f1.between(f2, f3, method)
                    betweennesses.append(btw)
                    value_dict[method].append(btw)
                except Exception:
                    betweennesses.append(float("NaN"))
                    value_dict[method].append(float("NaN"))
            print("{0}, {1}, {2}, {3}".format(first_fruit, second_fruit, third_fruit, ", ".join(map(lambda x: str(x), betweennesses))))

print("\n")

# histogramm data
print("lower_bound, upper_bound, {0}".format(", ".join(methods)))
num_bins = 10
bin_size = 1.0/num_bins

for i in range(num_bins + 1):
    counts = []
    lower_bound = i * bin_size
    upper_bound = (i + 1) * bin_size
    for method in methods:
        counts.append(len(filter(lambda x: lower_bound <= x < upper_bound, value_dict[method])))
    print("{0}, {1}, {2}".format(str(lower_bound), str(upper_bound), ", ".join(map(lambda x: str(x), counts))))