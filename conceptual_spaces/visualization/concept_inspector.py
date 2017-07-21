# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 15:30:59 2017

@author: lbechberger
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.widgets import RadioButtons, CheckButtons

import sys
sys.path.append("..")
import cs.cs as space

this = sys.modules[__name__]
this._dimensions = None
this._current_dims = None
this._concepts = None
this._active_concepts = None
this._plot_dim_indices = None
this._current_2d_indices = None
this._current_3d_indices = None
this._initialized = False
this._fig = None
this._ax3d = None
this._ax_2d = None

def _box_data_3d(p_min, p_max, indices):
    a = indices[0]
    b = indices[1]
    c = indices[2]    
    
    x = [[p_min[a], p_max[a], p_max[a], p_min[a], p_min[a]],  # bottom
         [p_min[a], p_max[a], p_max[a], p_min[a], p_min[a]],  # top
         [p_min[a], p_max[a], p_max[a], p_min[a], p_min[a]],  # front
         [p_min[a], p_max[a], p_max[a], p_min[a], p_min[a]]]  # back
         
    y = [[p_min[b], p_min[b], p_max[b], p_max[b], p_min[b]],  # bottom
         [p_min[b], p_min[b], p_max[b], p_max[b], p_min[b]],  # top
         [p_min[b], p_min[b], p_min[b], p_min[b], p_min[b]],  # front
         [p_max[b], p_max[b], p_max[b], p_max[b], p_max[b]]]  # back
         
    z = [[p_min[c], p_min[c], p_min[c], p_min[c], p_min[c]],  # bottom
         [p_max[c], p_max[c], p_max[c], p_max[c], p_max[c]],  # top
         [p_min[c], p_min[c], p_max[c], p_max[c], p_min[c]],  # front
         [p_min[c], p_min[c], p_max[c], p_max[c], p_min[c]]]  # back
    
    return x, y, z

# TODO: make path for overall concept, not individual cuboids
def _path_for_cuboid(p_min, p_max):
    verts = [p_min, [p_min[0], p_max[1]], p_max, [p_max[0], p_min[1]], p_min]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY]
    
    path = Path(verts, codes)
    return path

# TODO: make path for overall alpha-cut, not individual cuboids
def _path_for_cuboid_alpha_cut(p_min, p_max, epsilon_x, epsilon_y):
    verts = [[p_min[0] - epsilon_x, p_min[1]], [p_min[0] - epsilon_x, p_max[1]],
             [p_min[0], p_max[1] + epsilon_y], [p_max[0], p_max[1] + epsilon_y],
             [p_max[0] + epsilon_x, p_max[1]], [p_max[0] + epsilon_x, p_min[1]], 
             [p_max[0], p_min[1] - epsilon_y], [p_min[0], p_min[1] - epsilon_y], p_min]
    codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, 
             Path.LINETO, Path.LINETO, Path.LINETO, Path.LINETO,Path.CLOSEPOLY]
    
    path = Path(verts, codes)
    return path
    
def test_concepts():
    
    this._dimensions = ["hue", "round", "sweet"]
    this._current_dims = list(this._dimensions)
    this._plot_dim_indices = [(0,1), (0,2), (1,2)]
    this._current_2d_indices = list(this._plot_dim_indices)
    this._current_3d_indices = [0,1,2]
    
    # define some concepts    
    apple1 = ([0.5, 0.65, 0.35], [0.8, 0.8, 0.5])
    apple2 = ([0.65, 0.65, 0.4], [0.85, 0.8, 0.55])
    apple3 = ([0.7, 0.65, 0.45], [1.0, 0.8, 0.6])
    apple = [apple1, apple2, apple3]    
    
    banana1 = ([0.5, 0.1, 0.35], [0.75, 0.30, 0.55])
    banana2 = ([0.7, 0.1, 0.5], [0.8, 0.3, 0.7])
    banana3 = ([0.75, 0.1, 0.5], [0.85, 0.3, 1.00])
    banana = [banana1, banana2, banana3]    
    
    pear = [([0.5, 0.4, 0.35], [0.7, 0.6, 0.45])]

    orange = [([0.8, 0.9, 0.6], [0.9, 1.0, 0.7])]
    lemon = [([0.7, 0.45, 0.0], [0.8, 0.55, 0.1])]
    granny_smith = [([0.55, 0.70, 0.35], [0.6, 0.8, 0.45])]

    this._concepts = {"apple":(apple, 'r', (0.05, 0.10, 0.10)), "banana":(banana, 'y', (0.05, 0.02, 0.15)), 
                "pear":(pear, 'g', (0.05, 0.10, 0.10)), "orange":(orange, 'orange', (0.05, 0.05, 0.05)), 
                "lemon":(lemon, 'y', (0.05, 0.05, 0.01)), "Granny Smith":(granny_smith, 'g', (0.03, 0.03, 0.03))}    
    
    this._fig = plt.figure(figsize=(20,14))
    this._fig.canvas.set_window_title("ConceptInspector")

    # set up 3d plot    
    this._ax_3d = this._fig.add_subplot(221, projection='3d')
    def init_ax_3d():
        first_dim = this._current_dims[0]
        second_dim = this._current_dims[1]
        third_dim = this._current_dims[2]
        this._ax_3d.clear()
        this._ax_3d.set_title("3D visualization - {0}, {1}, {2}".format(first_dim, second_dim, third_dim))
        this._ax_3d.set_xlabel(first_dim)
        this._ax_3d.set_xlim(-0.1, 1.1)
        this._ax_3d.set_ylabel(second_dim)
        this._ax_3d.set_ylim(-0.1, 1.1)
        this._ax_3d.set_zlabel(third_dim)
        this._ax_3d.set_zlim(-0.1, 1.1)
    init_ax_3d()

    this._ax_2d = []

    # set up first 2d plot
    this._ax_2d.append(this._fig.add_subplot(222))
    def init_ax_2d(idx):
        first_dim = this._current_dims[this._plot_dim_indices[idx][0]]
        second_dim = this._current_dims[this._plot_dim_indices[idx][1]]
        this._ax_2d[idx].clear()
        this._ax_2d[idx].set_title("2D visualization - {0}, {1}".format(first_dim, second_dim))
        this._ax_2d[idx].set_xlabel(first_dim)
        this._ax_2d[idx].set_xlim(-0.2,1.2)
        this._ax_2d[idx].set_ylabel(second_dim)
        this._ax_2d[idx].set_ylim(-0.2,1.2)
    init_ax_2d(0)

    # set up second 2d plot
    this._ax_2d.append(this._fig.add_subplot(223))
    init_ax_2d(1)

    # set up third 2d plot
    this._ax_2d.append(this._fig.add_subplot(224))
    init_ax_2d(2)

    alpha_3d = 0.3    
    alpha_2d = 0.5    
    
    def draw_concept(concept, color, epsilons):
        for cuboid in concept:
            # plot cuboid in 3d
            x, y, z = _box_data_3d(cuboid[0], cuboid[1], this._current_3d_indices)
            this._ax_3d.plot_surface(x, y, z, color=color, rstride=1, cstride=1, alpha=alpha_3d)
            
            # plot cuboid in 2d projections
            for i in range(3):
                idx = this._current_2d_indices[i]
                cub_path = _path_for_cuboid([cuboid[0][idx[0]], cuboid[0][idx[1]]], [cuboid[1][idx[0]], cuboid[1][idx[1]]])
                cub_patch = patches.PathPatch(cub_path, facecolor=color, lw=2, alpha=alpha_2d)
                this._ax_2d[i].add_patch(cub_patch)
                
                alpha_path = _path_for_cuboid_alpha_cut([cuboid[0][idx[0]], cuboid[0][idx[1]]], [cuboid[1][idx[0]], cuboid[1][idx[1]]], epsilons[idx[0]], epsilons[idx[1]])
                alpha_patch = patches.PathPatch(alpha_path, facecolor='none', edgecolor=color, linestyle='dashed')
                this._ax_2d[i].add_patch(alpha_patch)
 
    
    this._active_concepts = list(this._concepts.keys())
    
    for concept_name in this._active_concepts:
        concept, color, epsilons = this._concepts[concept_name]
        draw_concept(concept, color, epsilons)
    
    # now add radio buttons for selecting dimensions
    this._fig.subplots_adjust(left=0.2, right=0.95)
    
    first_dim_radios_ax = this._fig.add_axes([0.025, 0.75, 0.1, 0.15], axisbg='w')
    first_dim_radios_ax.set_title("First dimension")
    first_dim_radios = RadioButtons(first_dim_radios_ax, this._dimensions, active=0)
    def first_dim_click_handler(label):
        this._current_dims[0] = label
        idx = this._dimensions.index(label)
        this._current_2d_indices[0] = (idx, this._current_2d_indices[0][1])
        this._current_2d_indices[1] = (idx, this._current_2d_indices[1][1])
        this._current_3d_indices[0] = idx
        init_ax_3d()
        init_ax_2d(0)
        init_ax_2d(1)
        init_ax_2d(2)
        for concept_name in this._active_concepts:
            concept, color, epsilons = this._concepts[concept_name]
            draw_concept(concept, color, epsilons)
        this._fig.canvas.draw_idle()
    first_dim_radios.on_clicked(first_dim_click_handler)
    
    second_dim_radios_ax = this._fig.add_axes([0.025, 0.55, 0.1, 0.15], axisbg='w')
    second_dim_radios_ax.set_title("Second dimension")
    second_dim_radios = RadioButtons(second_dim_radios_ax, this._dimensions, active=1)
    def second_dim_click_handler(label):
        this._current_dims[1] = label
        idx = this._dimensions.index(label)
        this._current_2d_indices[0] = (this._current_2d_indices[0][0], idx)
        this._current_2d_indices[2] = (idx, this._current_2d_indices[2][1])
        this._current_3d_indices[1] = idx
        init_ax_3d()
        init_ax_2d(0)
        init_ax_2d(1)
        init_ax_2d(2)
        for concept_name in this._active_concepts:
            concept, color, epsilons = this._concepts[concept_name]
            draw_concept(concept, color, epsilons)
        this._fig.canvas.draw_idle()
    second_dim_radios.on_clicked(second_dim_click_handler)

    third_dim_radios_ax = this._fig.add_axes([0.025, 0.35, 0.1, 0.15], axisbg='w')
    third_dim_radios_ax.set_title("Third dimension")
    third_dim_radios = RadioButtons(third_dim_radios_ax, this._dimensions, active=2)
    def third_dim_click_handler(label):
        this._current_dims[2] = label
        idx = this._dimensions.index(label)
        this._current_2d_indices[1] = (this._current_2d_indices[1][0], idx)
        this._current_2d_indices[2] = (this._current_2d_indices[2][0], idx)
        this._current_3d_indices[2] = idx
        init_ax_3d()
        init_ax_2d(0)
        init_ax_2d(1)
        init_ax_2d(2)
        for concept_name in this._active_concepts:
            concept, color, epsilons = this._concepts[concept_name]
            draw_concept(concept, color, epsilons)
        this._fig.canvas.draw_idle()
    third_dim_radios.on_clicked(third_dim_click_handler)
    
    # add check boxes for selecting concepts
    concept_checks_ax = this._fig.add_axes([0.025, 0.1, 0.1, 0.15], axisbg='w')
    concept_checks_ax.set_title("Concepts")
    concept_checks = CheckButtons(concept_checks_ax, list(this._active_concepts), [True]*6)
    c = map(lambda x: x[1], list(this._concepts.values()))    # color them nicely
    [rec.set_facecolor(c[i]) for i, rec in enumerate(concept_checks.rectangles)]
    def concept_checks_click_handler(label):
        if label in this._active_concepts:
            this._active_concepts.remove(label)
        else:
            this._active_concepts.append(label)
        init_ax_3d()
        init_ax_2d(0)
        init_ax_2d(1)
        init_ax_2d(2)
        for concept_name in this._active_concepts:
            concept, color, epsilons = this._concepts[concept_name]
            draw_concept(concept, color, epsilons)
        this._fig.canvas.draw_idle()
    concept_checks.on_clicked(concept_checks_click_handler)
        
    
    plt.show()

    return first_dim_radios, second_dim_radios, third_dim_radios, concept_checks
        
# MAIN:
x = test_concepts()