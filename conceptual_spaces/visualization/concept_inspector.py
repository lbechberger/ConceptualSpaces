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
import shapely.geometry

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
this._checks_ax = None
this._axis_ranges = None
this._alpha_2d = None
this._alpha_3d = None
this._initialized = False

def _cuboid_data_3d(p_min, p_max):
    """Returns the 3d information necessary for plotting a 3d cuboid."""
    
    a = this._current_3d_indices[0]
    b = this._current_3d_indices[1]
    c = this._current_3d_indices[2]    
    
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

def _path_for_core(cuboids, d1, d2):
    """Creates the 2d path for a complete core."""

    polygon = None    
    for cuboid in cuboids:
        p_min = cuboid[0]
        p_max = cuboid[1]
        cub = shapely.geometry.box(p_min[d1], p_min[d2], p_max[d1], p_max[d2])
        if polygon == None:
            polygon = cub
        else:
            polygon = polygon.union(cub)
    
    verts = list(polygon.exterior.coords)
    codes = [Path.LINETO] * len(verts)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    
    path = Path(verts, codes)
    return path

def _path_for_core_alpha_cut(cuboids, d1, d2, epsilon1, epsilon2):
    """Creates the 2d path for a concept's 0.5 cut."""    
    polygon = None
    for cuboid in cuboids:
        p_min = (cuboid[0][d1], cuboid[0][d2])
        p_max = (cuboid[1][d1], cuboid[1][d2])
        alphaCuboid = shapely.geometry.Polygon([[p_min[0] - epsilon1, p_min[1]], [p_min[0] - epsilon1, p_max[1]],
             [p_min[0], p_max[1] + epsilon2], [p_max[0], p_max[1] + epsilon2],
             [p_max[0] + epsilon1, p_max[1]], [p_max[0] + epsilon1, p_min[1]], 
             [p_max[0], p_min[1] - epsilon2], [p_min[0], p_min[1] - epsilon2], [p_min[0] - epsilon1, p_min[1]]])
        
        if polygon == None:
            polygon = alphaCuboid
        else:
            polygon = polygon.union(alphaCuboid)
    
    verts = list(polygon.exterior.coords)
    codes = [Path.LINETO] * len(verts)
    codes[0] = Path.MOVETO
    codes[-1] = Path.CLOSEPOLY
    
    path = Path(verts, codes)
    return path

def _init_ax_3d():
    """Sets up the three-dimensional plot with respect to labels and axis ranges."""
    
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
    """Sets up a two-dimensional plot with respect to labels and axis ranges."""
    
    first_dim = this._dimensions[this._current_2d_indices[idx][0]]
    second_dim = this._dimensions[this._current_2d_indices[idx][1]]
    this._ax_2d[idx].clear()
    this._ax_2d[idx].set_title("2D visualization - {0}, {1}".format(first_dim, second_dim))
    this._ax_2d[idx].set_xlabel(first_dim)
    this._ax_2d[idx].set_xlim(this._axis_ranges[this._current_2d_indices[idx][0]])
    this._ax_2d[idx].set_ylabel(second_dim)
    this._ax_2d[idx].set_ylim(this._axis_ranges[this._current_2d_indices[idx][1]])

def _draw_concept(concept, color, epsilons):
    """Paints a single concept into all four plots."""
    
    # plot cuboids separately in 3d
    for cuboid in concept:
        x, y, z = _cuboid_data_3d(cuboid[0], cuboid[1])
        this._ax_3d.plot_surface(x, y, z, color=color, rstride=1, cstride=1, alpha=this._alpha_3d)
    
    # plot overall cores in 2d
    for i in range(3):
        d1, d2 = this._current_2d_indices[i]
        core_path = _path_for_core(concept, d1, d2)
        core_patch = patches.PathPatch(core_path, facecolor=color, lw=2, alpha=this._alpha_2d)
        this._ax_2d[i].add_patch(core_patch)
        
        alpha_path = _path_for_core_alpha_cut(concept, d1, d2, epsilons[d1], epsilons[d2])
        alpha_patch = patches.PathPatch(alpha_path, facecolor='none', edgecolor=color, linestyle='dashed')
        this._ax_2d[i].add_patch(alpha_patch)

def _repaint_everything():
    """Repaints the whole window."""
    
    _init_ax_3d()
    _init_ax_2d(0)
    _init_ax_2d(1)
    _init_ax_2d(2)
    for concept_name in this._active_concepts:
        concept, color, epsilons = this._concepts[concept_name]
        _draw_concept(concept, color, epsilons)
    this._fig.canvas.draw_idle()

   
def init():
    """Initializes the ConceptInspector and displays it."""
    
    if this._initialized:   # make sure we can only call init once!
        return
    # figure out dimensionality of our space and initialize the plots accordingly
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

    # create the figure
    this._fig = plt.figure(figsize=(20,14))
    this._fig.canvas.set_window_title("ConceptInspector")

    # set up plots    
    this._ax_3d = this._fig.add_subplot(221, projection='3d')
    this._ax_2d = []
    this._ax_2d.append(this._fig.add_subplot(222))
    this._ax_2d.append(this._fig.add_subplot(223))
    this._ax_2d.append(this._fig.add_subplot(224))

    # set alpha parameters
    this._alpha_3d = 0.2    
    this._alpha_2d = 0.4    
    
    # now add radio buttons for selecting dimensions
    this._fig.subplots_adjust(left=0.2, right=0.98, top=0.95, bottom=0.05)
    
    first_dim_radios_ax = this._fig.add_axes([0.025, 0.95 - 0.03 * space._n_dim, 0.12, 0.03 * space._n_dim], axisbg='w') 
    first_dim_radios_ax.set_title("First dimension")
    first_dim_radios = RadioButtons(first_dim_radios_ax, this._dimensions, active=0)
    def first_dim_click_handler(label):
        idx = this._dimensions.index(label)
        this._current_2d_indices[0] = (idx, this._current_2d_indices[0][1])
        this._current_2d_indices[1] = (idx, this._current_2d_indices[1][1])
        this._current_3d_indices[0] = idx
        _repaint_everything()
    first_dim_radios.on_clicked(first_dim_click_handler)
    
    second_dim_radios_ax = this._fig.add_axes([0.025, 0.95 - 0.06 * space._n_dim - 0.05, 0.12, 0.03 * space._n_dim], axisbg='w')
    second_dim_radios_ax.set_title("Second dimension")
    second_dim_radios = RadioButtons(second_dim_radios_ax, this._dimensions, active=1)
    def second_dim_click_handler(label):
        idx = this._dimensions.index(label)
        this._current_2d_indices[0] = (this._current_2d_indices[0][0], idx)
        this._current_2d_indices[2] = (idx, this._current_2d_indices[2][1])
        this._current_3d_indices[1] = idx
        _repaint_everything()
        this._fig.canvas.draw_idle()
    second_dim_radios.on_clicked(second_dim_click_handler)

    third_dim_radios_ax = this._fig.add_axes([0.025, 0.95 - 0.09 * space._n_dim - 0.10, 0.12, 0.03 * space._n_dim], axisbg='w')
    third_dim_radios_ax.set_title("Third dimension")
    third_dim_radios = RadioButtons(third_dim_radios_ax, this._dimensions, active=2)
    def third_dim_click_handler(label):
        idx = this._dimensions.index(label)
        this._current_2d_indices[1] = (this._current_2d_indices[1][0], idx)
        this._current_2d_indices[2] = (this._current_2d_indices[2][0], idx)
        this._current_3d_indices[2] = idx
        _repaint_everything()
    third_dim_radios.on_clicked(third_dim_click_handler)

    this._radios = (first_dim_radios, second_dim_radios, third_dim_radios)    
    
    # add area for check boxes (concept selection)
    this._checks_ax = this._fig.add_axes([0.025, 0.05, 0.12, 0.15], axisbg='w')

    # load all concepts, draw everything, then display the window    
    this._initialized = True
    update()
    plt.show()


def update():
    """Updates the list of concepts based on the dictionary of concepts stored in the cs module. Repaints everything. """
    
    if not this._initialized:   # if the figure is not initialized, yet, we cannot update it
        return
    # set up the range for each of the dimensions
    this._axis_ranges = [(float('inf'), -float('inf'))] * space._n_dim
    for concept in space._concepts.values():
        for cuboid in concept._core._cuboids:
            this._axis_ranges = list(map(lambda x, y: (min(x[0], y), x[1]) if not isinf(y) else x, this._axis_ranges, cuboid._p_min))
            this._axis_ranges = list(map(lambda x, y: (x[0], max(x[1], y)) if not isinf(y) else x, this._axis_ranges, cuboid._p_max))
    # add a little space in each dimensions for the plots
    widths = list(map(lambda x: x[1] - x[0], this._axis_ranges))
    this._axis_ranges = map(lambda x, y: (x[0] - 0.1 * y, x[1] + 0.1 * y), this._axis_ranges, widths)

    # grab all concepts
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

    # in the beginning, all concepts are active
    if this._active_concepts == None:    
        this._active_concepts = list(this._concepts.keys())

    # remove deleted concepts from our active list
    for name in this._active_concepts:
        if not name in this._concepts:
            this._active_concepts.remove(name)
    
    # recreate the check boxes
    this._checks_ax.clear()
    this._checks_ax.set_position([0.025, 0.95 - 0.09 * space._n_dim - 0.15 - 0.03 * len(this._concepts.keys()), 0.12, 0.03 * len(this._concepts.keys())])
    this._checks_ax.set_title("Concepts")
    this._checks = CheckButtons(this._checks_ax, list(sorted(this._concepts.keys())), list(map(lambda x: x in this._active_concepts, list(sorted(this._concepts.keys())))))
    c = list(map(lambda x: this._concepts[x][1], list(sorted(this._concepts.keys()))))    # color them nicely
    [rec.set_facecolor(c[i]) for i, rec in enumerate(this._checks.rectangles)]
    def concept_checks_click_handler(label):
        if label in this._active_concepts:
            this._active_concepts.remove(label)
        else:
            this._active_concepts.append(label)
        _repaint_everything()
    this._checks.on_clicked(concept_checks_click_handler)
    
    _repaint_everything()    