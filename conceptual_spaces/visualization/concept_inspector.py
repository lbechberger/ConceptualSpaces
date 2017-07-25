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
from collections import deque
from math import log, sqrt, isinf

import sys
sys.path.append("..")
import cs.cs as space

this = sys.modules[__name__]
this._dimensions = None
this._concepts = None
this._active_concepts = None
this._current_2d_indices = None
this._current_3d_indices = None
this._fig = None
this._ax3d = None
this._ax_2d = None
this._radios = None
this._checks = None
this._axis_ranges = None
this._initialized = False

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

def _init_ax_3d():
    first_dim = this._dimensions[this._current_3d_indices[0]]
    second_dim = this._dimensions[this._current_3d_indices[1]]
    third_dim = this._dimensions[this._current_3d_indices[2]]
    this._ax_3d.clear()
    this._ax_3d.set_title("3D visualization - {0}, {1}, {2}".format(first_dim, second_dim, third_dim))
    this._ax_3d.set_xlabel(first_dim)
    this._ax_3d.set_xlim(this._axis_ranges[this._current_3d_indices[0]])
    this._ax_3d.set_ylabel(second_dim)
    this._ax_3d.set_ylim(this._axis_ranges[this._current_3d_indices[1]])
    this._ax_3d.set_zlabel(third_dim)
    this._ax_3d.set_zlim(this._axis_ranges[this._current_3d_indices[2]])

def _init_ax_2d(idx):
    first_dim = this._dimensions[this._current_2d_indices[idx][0]]
    second_dim = this._dimensions[this._current_2d_indices[idx][1]]
    this._ax_2d[idx].clear()
    this._ax_2d[idx].set_title("2D visualization - {0}, {1}".format(first_dim, second_dim))
    this._ax_2d[idx].set_xlabel(first_dim)
    this._ax_2d[idx].set_xlim(this._axis_ranges[this._current_2d_indices[idx][0]])
    this._ax_2d[idx].set_ylabel(second_dim)
    this._ax_2d[idx].set_ylim(this._axis_ranges[this._current_2d_indices[idx][1]])
    
def init():
    
    this._dimensions = list(space._dim_names)
    if len(this._dimensions) >= 3:
        this._current_2d_indices = [(0,1), (0,2), (1,2)]
        this._current_3d_indices = [0,1,2]
    elif len(this._dimensions) == 2:
        this._current_2d_indices = [(0,1), (1,0), (0,1)]
        this._current_3d_indices = [0,1,1]
    else:
        this._current_2d_indices = [(0,0), (0,0), (0,0)]
        this._current_3d_indices = [0,0,0]

    # set up the range for each of the dimensions
    this._axis_ranges = [(float('inf'), -float('inf'))] * space._n_dim
    for concept in space._concepts.values():
        for cuboid in concept._core._cuboids:
            this._axis_ranges = list(map(lambda x, y: (min(x[0], y), x[1]) if not isinf(y) else x, this._axis_ranges, cuboid._p_min))
            this._axis_ranges = list(map(lambda x, y: (x[0], max(x[1], y)) if not isinf(y) else x, this._axis_ranges, cuboid._p_max))
    # add a little space in each dimensions for the plots
    widths = list(map(lambda x: x[1] - x[0], this._axis_ranges))
    this._axis_ranges = map(lambda x, y: (x[0] - 0.1 * y, x[1] + 0.1 * y), this._axis_ranges, widths)
    
    # load the concepts  
    standard_colors = deque(['r', 'g', 'b', 'y', 'purple', 'orange', 'brown', 'gray'])
    this._concepts = {}
    for name in space._concepts.keys():
        concept = space._concepts[name]
        color = None
        # take prespecified color or use next one in list
        if name in space._concept_colors:
            color = space._concept_colors[name]
        else:
            color = standard_colors.popleft()
            standard_colors.append(color)

        # collect all cuboids and replace infinities with boundaries of the plot
        cuboids = list(map(lambda x: (map(lambda y, z: max(y, z[0] - 0.01), x._p_min, this._axis_ranges), 
                                      map(lambda y, z: min(y, z[1] + 0.01), x._p_max, this._axis_ranges)), concept._core._cuboids))
        
        epsilons = []
        for dim in range(space._n_dim):
            eps = - (1.0 / concept._c) * log(0.5 / concept._mu)
            w_dim = None
            w_dom = None
            for dom, dims in concept._core._domains.items():
                if dim in dims:
                    w_dom = concept._weights._domain_weights[dom]
                    w_dim = concept._weights._dimension_weights[dom][dim]
                    break
            if w_dim == None or w_dom == None:
                eps = 0
            else:
                eps = eps / (w_dom * sqrt(w_dim))
            epsilons.append(eps)
        this._concepts[name] = (cuboids, color, epsilons)

    # create the figure
    this._fig = plt.figure(figsize=(20,14))
    this._fig.canvas.set_window_title("ConceptInspector")

    # set up 3d plot    
    this._ax_3d = this._fig.add_subplot(221, projection='3d')
    _init_ax_3d()

    # set up 2d plots
    this._ax_2d = []
    this._ax_2d.append(this._fig.add_subplot(222))
    _init_ax_2d(0)
    this._ax_2d.append(this._fig.add_subplot(223))
    _init_ax_2d(1)
    this._ax_2d.append(this._fig.add_subplot(224))
    _init_ax_2d(2)

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
    this._fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.05)
    
    first_dim_radios_ax = this._fig.add_axes([0.025, 0.75, 0.1, 0.15], axisbg='w')
    first_dim_radios_ax.set_title("First dimension")
    first_dim_radios = RadioButtons(first_dim_radios_ax, this._dimensions, active=0)
    def first_dim_click_handler(label):
        idx = this._dimensions.index(label)
        this._current_2d_indices[0] = (idx, this._current_2d_indices[0][1])
        this._current_2d_indices[1] = (idx, this._current_2d_indices[1][1])
        this._current_3d_indices[0] = idx
        _init_ax_3d()
        _init_ax_2d(0)
        _init_ax_2d(1)
        _init_ax_2d(2)
        for concept_name in this._active_concepts:
            concept, color, epsilons = this._concepts[concept_name]
            draw_concept(concept, color, epsilons)
        this._fig.canvas.draw_idle()
    first_dim_radios.on_clicked(first_dim_click_handler)
    
    second_dim_radios_ax = this._fig.add_axes([0.025, 0.55, 0.1, 0.15], axisbg='w')
    second_dim_radios_ax.set_title("Second dimension")
    second_dim_radios = RadioButtons(second_dim_radios_ax, this._dimensions, active=1)
    def second_dim_click_handler(label):
        idx = this._dimensions.index(label)
        this._current_2d_indices[0] = (this._current_2d_indices[0][0], idx)
        this._current_2d_indices[2] = (idx, this._current_2d_indices[2][1])
        this._current_3d_indices[1] = idx
        _init_ax_3d()
        _init_ax_2d(0)
        _init_ax_2d(1)
        _init_ax_2d(2)
        for concept_name in this._active_concepts:
            concept, color, epsilons = this._concepts[concept_name]
            draw_concept(concept, color, epsilons)
        this._fig.canvas.draw_idle()
    second_dim_radios.on_clicked(second_dim_click_handler)

    third_dim_radios_ax = this._fig.add_axes([0.025, 0.35, 0.1, 0.15], axisbg='w')
    third_dim_radios_ax.set_title("Third dimension")
    third_dim_radios = RadioButtons(third_dim_radios_ax, this._dimensions, active=2)
    def third_dim_click_handler(label):
        idx = this._dimensions.index(label)
        this._current_2d_indices[1] = (this._current_2d_indices[1][0], idx)
        this._current_2d_indices[2] = (this._current_2d_indices[2][0], idx)
        this._current_3d_indices[2] = idx
        _init_ax_3d()
        _init_ax_2d(0)
        _init_ax_2d(1)
        _init_ax_2d(2)
        for concept_name in this._active_concepts:
            concept, color, epsilons = this._concepts[concept_name]
            draw_concept(concept, color, epsilons)
        this._fig.canvas.draw_idle()
    third_dim_radios.on_clicked(third_dim_click_handler)

    this._radios = (first_dim_radios, second_dim_radios, third_dim_radios)    
    
    # add check boxes for selecting concepts
    concept_checks_ax = this._fig.add_axes([0.025, 0.1, 0.1, 0.15], axisbg='w')
    concept_checks_ax.set_title("Concepts")
    concept_checks = CheckButtons(concept_checks_ax, list(this._active_concepts), [True]*len(this._concepts.keys()))
    c = map(lambda x: x[1], list(this._concepts.values()))    # color them nicely
    [rec.set_facecolor(c[i]) for i, rec in enumerate(concept_checks.rectangles)]
    def concept_checks_click_handler(label):
        if label in this._active_concepts:
            this._active_concepts.remove(label)
        else:
            this._active_concepts.append(label)
        _init_ax_3d()
        _init_ax_2d(0)
        _init_ax_2d(1)
        _init_ax_2d(2)
        for concept_name in this._active_concepts:
            concept, color, epsilons = this._concepts[concept_name]
            draw_concept(concept, color, epsilons)
        this._fig.canvas.draw_idle()
    concept_checks.on_clicked(concept_checks_click_handler)
    
    this._checks = concept_checks
    
    plt.show()