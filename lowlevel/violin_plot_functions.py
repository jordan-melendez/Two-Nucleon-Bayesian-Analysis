from __future__ import division
from textwrap import dedent
import colorsys
import numpy as np
from numpy import vectorize, fmax
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import fmin
import pandas as pd
from pandas.core.series import remove_na
import matplotlib as mpl
from matplotlib.collections import PatchCollection
from matplotlib.ticker import MultipleLocator, LinearLocator, IndexLocator
import matplotlib.patches as Patches
import matplotlib.pyplot as plt
import warnings

from seaborn.external.six import string_types
from seaborn.external.six.moves import range

from seaborn import utils
from seaborn.utils import iqr, categorical_order
from seaborn.algorithms import bootstrap
from seaborn.palettes import color_palette, husl_palette, light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs

from CH_to_EKM_statistics import *


__all__ = ["violinfunctionplot"]


class _CategoricalFunctionPlotter(object):

    width = .8

    def establish_variables(self, x=None, y=None, category=None, hue=None,
                            data=None, onesided=False, orient=None, order=None,
                            hue_order=None, units=None):
        """Convert input specification into a common representation."""
        # Option 1:
        # We are plotting a wide-form dataset
        # -----------------------------------
        if x is None and y is None:

            # I don't think the default code works in my 'function' case.
            error = "An x and y must be given"
            raise ValueError(error)

            # All the rest doesn't matter

            # Do a sanity check on the inputs
            if hue is not None:
                error = "Cannot use `hue` without `x` or `y`"
                raise ValueError(error)

            # No hue grouping with wide inputs
            plot_hues = None
            hue_title = None
            hue_names = None

            # No statistical units with wide inputs
            plot_units = None

            # We also won't get axes labels here
            value_label = None
            group_label = None

            # Option 1a:
            # The input data is a Pandas DataFrame
            # ------------------------------------

            if isinstance(data, pd.DataFrame):

                # Order the data correctly
                if order is None:
                    order = []
                    # Reduce to just numeric columns
                    for col in data:
                        try:
                            data[col].astype(np.float)
                            order.append(col)
                        except ValueError:
                            pass
                # ind_var = data[ivar]
                plot_data = data[order]
                group_names = order
                group_label = data.columns.name

                # Convert to a list of arrays, the common representation
                iter_data = plot_data.iteritems()
                plot_data = [np.asarray(s, np.float) for k, s in iter_data]

            # Option 1b:
            # The input data is an array or list
            # ----------------------------------

            else:

                # We can't reorder the data
                if order is not None:
                    error = "Input data must be a pandas object to reorder"
                    raise ValueError(error)

                # The input data is an array
                if hasattr(data, "shape"):
                    if len(data.shape) == 1:
                        if np.isscalar(data[0]):
                            plot_data = [data]
                        else:
                            plot_data = list(data)
                    elif len(data.shape) == 2:
                        nr, nc = data.shape
                        if nr == 1 or nc == 1:
                            plot_data = [data.ravel()]
                        else:
                            plot_data = [data[:, i] for i in range(nc)]
                    else:
                        error = ("Input `data` can have no "
                                 "more than 2 dimensions")
                        raise ValueError(error)

                # Check if `data` is None to let us bail out here (for testing)
                elif data is None:
                    plot_data = [[]]

                # The input data is a flat list
                elif np.isscalar(data[0]):
                    plot_data = [data]

                # The input data is a nested list
                # This will catch some things that might fail later
                # but exhaustive checks are hard
                else:
                    plot_data = data

                # Convert to a list of arrays, the common representation
                plot_data = [np.asarray(d, np.float) for d in plot_data]

                # The group names will just be numeric indices
                group_names = list(range((len(plot_data))))

            # Figure out the plotting orientation
            orient = "h" if str(orient).startswith("h") else "v"

        # Option 2:
        # We are plotting a long-form dataset
        # -----------------------------------

        else:

            # See if we need to get variables from `data`
            if data is not None:
                # print(data)
                # print(x)
                x = data.get(x, x)
                y = data.get(y, y)
                category = data.get(category, category)
                hue = data.get(hue, hue)
                units = data.get(units, units)

            # Validate the inputs
            for input in [x, y, category, hue, units]:
                if isinstance(input, string_types):
                    err = "Could not interpret input '{}'".format(input)
                    raise ValueError(err)

            # Figure out the plotting orientation
            # orient = self.infer_orient(x, y, orient)

            # Option 2a:
            # We are plotting a single set of data
            # ------------------------------------
            if x is None or y is None:

                # I don't think the default code works in my 'function' case.
                error = "An x and y must be given"
                raise ValueError(error)

                # All the rest doesn't matter

                # Determine where the data are
                vals = y if x is None else x

                # Put them into the common representation
                plot_data = [np.asarray(vals)]

                # Get a label for the value axis
                if hasattr(vals, "name"):
                    value_label = vals.name
                else:
                    value_label = None

                # This plot will not have group labels or hue nesting
                groups = None
                group_label = None
                group_names = []
                plot_hues = None
                hue_names = None
                hue_title = None
                plot_units = None

            # Option 2b:
            # We are grouping the data values by another variable
            # ---------------------------------------------------
            else:

                # Determine which role each variable will play
                if orient == "v":
                    # vals, groups = y, x
                    ind_var, vals = y, x
                else:
                    # vals, groups = x, y
                    ind_var, vals = x, y

                ind_var_label = ind_var.name

                groups = category
                # Get the categorical axis label
                group_label = None
                if hasattr(groups, "name"):
                    group_label = groups.name

                # Get the order on the categorical axis
                group_names = categorical_order(groups, order)

                # Group the independent variable data
                ind_var_data, ind_var_label = self._group_longform(
                    ind_var, groups, group_names
                    )

                # Group the function data
                plot_data, value_label = self._group_longform(vals, groups,
                                                              group_names)

                # Now handle the hue levels for nested ordering
                if hue is None:
                    plot_hues = None
                    hue_title = None
                    hue_names = None
                else:

                    # Get the order of the hue levels
                    hue_names = categorical_order(hue, hue_order)

                    # Group the hue data
                    plot_hues, hue_title = self._group_longform(hue, groups,
                                                                group_names)

                # Now handle the units for nested observations
                if units is None:
                    plot_units = None
                else:
                    plot_units, _ = self._group_longform(units, groups,
                                                         group_names)

        # Assign object attributes
        # ------------------------
        self.orient = orient
        self.plot_data = plot_data
        self.group_label = group_label
        self.value_label = value_label
        self.group_names = group_names
        self.plot_hues = plot_hues
        self.hue_title = hue_title
        self.hue_names = hue_names
        self.plot_units = plot_units
        self.onesided = onesided

        # Function changes
        self.ind_var_label = ind_var_label
        self.ind_var_data = ind_var_data
        # Rename stuff too for later just in case
        # self.support = ind_var_data
        # self.density = plot_data

        if self.hue_names is None:
            # support = []
            # density = []
            # counts = np.zeros(len(self.plot_data))
            max_density = np.zeros(len(self.plot_data))
            self.support = ind_var_data
            self.density = plot_data
            for i, group_data in enumerate(self.plot_data):
                max_density[i] = group_data.max()
        else:
            support = [[] for _ in self.plot_data]
            density = [[] for _ in self.plot_data]
            size = len(self.group_names), len(self.hue_names)
            # counts = np.zeros(size)
            max_density = np.zeros(size)
            for i, (ivar_data, group_data) in enumerate(
                            zip(self.ind_var_data, self.plot_data)):
                for j, hue_level in enumerate(self.hue_names):

                        # Select out the observations for this hue level
                        hue_mask = self.plot_hues[i] == hue_level

                        # Strip missing datapoints
                        ivar_data_ij = remove_na(ivar_data[hue_mask])
                        group_data_ij = remove_na(group_data[hue_mask])

                        # Determine the support grid and get the density over it
                        # support_ij = self.kde_support(kde_data, bw_used,
                        #                               cut, gridsize)
                        # density_ij = kde.evaluate(support_ij)

                        # Update the data structures with these results
                        support[i].append(ivar_data_ij)
                        density[i].append(group_data_ij)
                        # counts[i, j] = kde_data.size
                        # max_density[i, j] = density_ij.max()
                        max_density[i, j] = group_data_ij.max()

            self.support = support
            self.density = density
            self.max_density = max_density

    def _group_longform(self, vals, grouper, order):
        """Group a long-form variable by another with correct order."""
        # Ensure that the groupby will work
        if not isinstance(vals, pd.Series):
            vals = pd.Series(vals)

        # Group the val data
        grouped_vals = vals.groupby(grouper)
        out_data = []
        for g in order:
            try:
                g_vals = np.asarray(grouped_vals.get_group(g))
            except KeyError:
                g_vals = np.array([])
            out_data.append(g_vals)

        # Get the vals axis label
        label = vals.name

        return out_data, label

    def establish_colors(self, color, palette, saturation):
        """Get a list of colors for the main component of the plots."""
        if self.hue_names is None:
            n_colors = len(self.plot_data)
        else:
            n_colors = len(self.hue_names)

        # Determine the main colors
        if color is None and palette is None:
            # Determine whether the current palette will have enough values
            # If not, we'll default to the husl palette so each is distinct
            current_palette = utils.get_color_cycle()
            if n_colors <= len(current_palette):
                colors = color_palette(n_colors=n_colors)
            else:
                colors = husl_palette(n_colors, l=.7)

        elif palette is None:
            # When passing a specific color, the interpretation depends
            # on whether there is a hue variable or not.
            # If so, we will make a blend palette so that the different
            # levels have some amount of variation.
            if self.hue_names is None:
                colors = [color] * n_colors
            else:
                colors = light_palette(color, n_colors)
        else:

            # Let `palette` be a dict mapping level to color
            if isinstance(palette, dict):
                if self.hue_names is None:
                    levels = self.group_names
                else:
                    levels = self.hue_names
                palette = [palette[l] for l in levels]

            colors = color_palette(palette, n_colors)

        # Desaturate a bit because these are patches
        if saturation < 1:
            colors = color_palette(colors, desat=saturation)

        # Conver the colors to a common representations
        rgb_colors = color_palette(colors)

        # Determine the gray color to use for the lines framing the plot
        light_vals = [colorsys.rgb_to_hls(*c)[1] for c in rgb_colors]
        l = min(light_vals) * .6
        gray = mpl.colors.rgb2hex((l, l, l))

        # Assign object attributes
        self.colors = rgb_colors
        self.gray = gray

    def infer_orient(self, x, y, orient=None):
        """Determine how the plot should be oriented based on the data."""
        orient = str(orient)

        def is_categorical(s):
            try:
                # Correct way, but doesnt exist in older Pandas
                return pd.core.common.is_categorical_dtype(s)
            except AttributeError:
                # Also works, but feels hackier
                return str(s.dtype) == "categorical"

        def is_not_numeric(s):
            try:
                np.asarray(s, dtype=np.float)
            except ValueError:
                return True
            return False

        no_numeric = "Neither the `x` nor `y` variable appears to be numeric."

        if orient.startswith("v"):
            return "v"
        elif orient.startswith("h"):
            return "h"
        elif x is None:
            return "v"
        elif y is None:
            return "h"
        elif is_categorical(y):
            if is_categorical(x):
                raise ValueError(no_numeric)
            else:
                return "h"
        elif is_not_numeric(y):
            if is_not_numeric(x):
                raise ValueError(no_numeric)
            else:
                return "h"
        else:
            return "v"

    @property
    def hue_offsets(self):
        """A list of center positions for plots when hue nesting is used."""
        n_levels = len(self.hue_names)
        # if self.onesided:
        #     each_width = 2*self.width / (n_levels)
        #     offsets = np.linspace(0, self.width/2 - each_width, n_levels)
        #     offsets -= offsets.mean()
        if self.dodge:
            each_width = self.width / n_levels
            offsets = np.linspace(0, self.width - each_width, n_levels)
            offsets -= offsets.mean()
        else:
            offsets = np.zeros(n_levels)

        return offsets

    @property
    def nested_width(self):
        """A float with the width of plot elements when hue nesting is used."""

        if self.dodge:
            width = self.width / len(self.hue_names) * .98
        else:
            width = self.width
        return width

    def annotate_axes(self, ax):
        """Add descriptive labels to an Axes object."""
        if self.orient == "v":
            xlabel, ylabel = self.group_label, self.ind_var_label
        else:
            xlabel, ylabel = self.ind_var_label, self.group_label

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)

        if self.orient == "v":
            ax.set_xticks(np.arange(len(self.plot_data)))
            ax.set_xticklabels(self.group_names)
        else:
            ax.set_yticks(np.arange(len(self.plot_data)))
            ax.set_yticklabels(self.group_names)

        if self.orient == "v":
            ax.xaxis.grid(False)
            ax.set_xlim(-.5, len(self.plot_data) - .5)
        else:
            ax.yaxis.grid(False)
            # minorLocator = MultipleLocator(0.5)
            minorLocator = IndexLocator(1, 0)
            # ax.yaxis.set_minor_locator(minorLocator)
            ax.set_yticks(np.arange(len(self.plot_data))+self.width/2, minor=True)
            if self.onesided:
                ax.set_ylim(-.5/self.width, len(self.plot_data) + -.5/self.width)
            else:
                ax.set_ylim(-.5, len(self.plot_data) - .5)

        if self.hue_names is not None:
            try:
                leg_text_size = mpl.rcParams["axes.labelsize"] * .8
            except TypeError:  # labelsize is something like "large"
                leg_text_size = mpl.rcParams["axes.labelsize"]

            leg = ax.legend(loc="best", frameon=True, fancybox=True,
                            fontsize=leg_text_size)
            if self.hue_title is not None:
                leg.set_title(self.hue_title)

                # Set the title size a roundabout way to maintain
                # compatability with matplotlib 1.1
                try:
                    title_size = mpl.rcParams["axes.labelsize"] * .8
                except TypeError:  # labelsize is something like "large"
                    title_size = mpl.rcParams["axes.labelsize"]
                prop = mpl.font_manager.FontProperties(size=title_size)
                leg._legend_title_box._text.set_font_properties(prop)

    def add_legend_data(self, ax, color, label):
        """Add a dummy patch object so we can get legend data."""
        rect = plt.Rectangle([0, 0], 0, 0,
                             linewidth=self.linewidth / 2,
                             edgecolor=self.gray,
                             facecolor=color,
                             label=label)
        ax.add_patch(rect)


class _ViolinFunctionPlotter(_CategoricalFunctionPlotter):

    def __init__(self, x, y, category, hue, data, order, hue_order,
                 # bw, cut,
                 scale, scale_hue, gridsize,
                 width, inner, split, dodge, onesided, orient, linewidth,
                 color, palette, saturation, HDI):

        self.establish_variables(x, y, category, hue, data, onesided, orient,
                                 order, hue_order)
        self.establish_colors(color, palette, saturation)
        # self.estimate_densities(bw, cut, scale, scale_hue, gridsize)

        # self.get_percentiles(support, density)

        # if scale == "area":
        #     self.scale_area(self.density, self.max_density, scale_hue)
        # elif scale == "width":
        #     self.scale_width(self.density)

        if scale == "area":
            self.density_scale = self.max_density.max() * np.ones(self.max_density.shape)
        elif scale == "width":
            self.density_scale = self.max_density
        # print(self.density_scale)

        # Use highest density intervals?
        self.HDI = HDI

        # self.gridsize = gridsize
        if self.hue_names is None:
            self.gridsize = len(self.support[0])
        else:
            self.gridsize = len(self.support[0][0])
        self.width = width
        self.dodge = dodge

        if inner is not None:
            if not any([
                        # inner.startswith("quart"),
                        inner.startswith("center"),
                        inner.startswith("orth")
                        # inner.startswith("stick"),
                        # inner.startswith("point")
                        ]):
                err = "Inner style '{}' not recognized".format(inner)
                raise ValueError(err)
        self.inner = inner

        if split and self.hue_names is not None and len(self.hue_names) != 2:
            raise ValueError("Cannot use `split` with more than 2 hue levels.")
        self.split = split

        if linewidth is None:
            linewidth = mpl.rcParams["lines.linewidth"]
        self.linewidth = linewidth

    def scale_area(self, density, max_density, scale_hue):
        """Scale the relative area under the KDE curve.
        This essentially preserves the "standard" KDE scaling, but the
        resulting maximum density will be 1 so that the curve can be
        properly multiplied by the violin width.
        """
        if self.hue_names is None:
            for d in density:
                if d.size > 1:
                    d /= max_density.max()
        else:
            for i, group in enumerate(density):
                for d in group:
                    if scale_hue:
                        max = max_density[i].max()
                    else:
                        max = max_density.max()
                    if d.size > 1:
                        d /= max

    def scale_width(self, density):
        """Scale each density curve to the same height."""
        if self.hue_names is None:
            for d in density:
                d /= d.max()
        else:
            for group in density:
                for d in group:
                    d /= d.max()

    @property
    def dwidth(self):

        if self.hue_names is None or not self.dodge:
            return self.width / 2
        elif self.split:
            return self.width / 2
        elif self.onesided:
            return self.width / 2
        else:
            return self.width / (2 * len(self.hue_names))

    def draw_violins(self, ax):
        """Draw the violins onto `ax`."""
        fill_func = ax.fill_betweenx if self.orient == "v" else ax.fill_between
        kws = dict(edgecolor=self.gray, linewidth=self.linewidth/2, zorder=2, linestyle='-')

        for i, group_data in enumerate(self.plot_data):

            # Option 1: we have a single level of grouping
            # --------------------------------------------

            if self.plot_hues is None:

                support, density, max_density = self.support[i], self.density[i], self.density_scale[i]

                plot_support = np.array([s for s, d in zip(support, density) if d/max_density >= 0.01])
                plot_density = np.array([d for d in density if d/max_density >= 0.01])

                # # Handle special case of no observations in this bin
                # if support.size == 0:
                #     continue

                # # Handle special case of a single observation
                # elif support.size == 1:
                #     val = np.asscalar(support)
                #     d = np.asscalar(density)
                #     self.draw_single_observation(ax, i, val, d)
                #     continue

                # Draw the violin for this group
                # grid = np.ones(self.gridsize) * i
                grid = np.ones(len(plot_density)) * i
                fill_func(plot_support,
                          grid - plot_density * self.dwidth / max_density,
                          grid + plot_density * self.dwidth / max_density,
                          facecolor=self.colors[i],
                          **kws)

                # Draw the interior representation of the data
                if self.inner is None:
                    continue

                # Get a nan-free vector of datapoints
                violin_data = remove_na(group_data)

                # Draw 68-95% lines down center of violin
                if self.inner.startswith("center"):
                    self.draw_box_lines(ax, violin_data, support, density, i, self.HDI)

                # Draw 68-95% lines orthogonal to violin
                elif self.inner.startswith("orth"):
                    self.draw_quartiles(ax, violin_data, support, density, max_density, i)

            # Option 2: we have nested grouping by a hue variable
            # ---------------------------------------------------

            else:
                offsets = self.hue_offsets
                for j, hue_level in enumerate(self.hue_names):

                    support, density = self.support[i][j], self.density[i][j]
                    max_density = self.density_scale[i][j]

                    plot_support = np.array([s for s, d in zip(support, density) if d/max_density >= 0.01])
                    plot_density = np.array([d for d in density if d/max_density >= 0.01])
                    # print(support)
                    # print(density)
                    kws["color"] = self.colors[j]

                    # Add legend data, but just for one set of violins
                    if not i:
                        self.add_legend_data(ax, self.colors[j], hue_level)

                    # # Handle the special case where we have no observations
                    # if support.size == 0:
                    #     continue

                    # # Handle the special case where we have one observation
                    # elif support.size == 1:
                    #     val = np.asscalar(support)
                    #     d = np.asscalar(density)
                    #     if self.split:
                    #         d = d / 2
                    #     at_group = i + offsets[j]
                    #     self.draw_single_observation(ax, at_group, val, d)
                    #     continue

                    # Option 2a: we are drawing a single split violin
                    # -----------------------------------------------

                    if self.split:

                        # grid = np.ones(self.gridsize) * i
                        grid = np.ones(len(plot_density)) * i
                        if j:
                            fill_func(plot_support,
                                      grid,
                                      grid + plot_density * self.dwidth / max_density,
                                      **kws)
                            # Sometimes first one doesn't draw gray outline
                            fill_func(plot_support,
                                      grid,
                                      grid + plot_density * self.dwidth / max_density,
                                      **kws)
                            fill_func(plot_support,
                                      grid,
                                      grid + plot_density * self.dwidth / max_density,
                                      **kws)
                        else:
                            fill_func(plot_support,
                                      grid - plot_density * self.dwidth / max_density,
                                      grid,
                                      **kws)
                            # Sometimes first one doesn't draw gray outline
                            fill_func(plot_support,
                                      grid - plot_density * self.dwidth / max_density,
                                      grid,
                                      **kws)
                            fill_func(plot_support,
                                      grid - plot_density * self.dwidth / max_density,
                                      grid,
                                      **kws)

                        # Draw the interior representation of the data
                        if self.inner is None:
                            continue

                        # Get a nan-free vector of datapoints
                        hue_mask = self.plot_hues[i] == hue_level
                        violin_data = remove_na(group_data[hue_mask])

                        # # Draw quartile lines
                        # if self.inner.startswith("quart"):
                        #     self.draw_quartiles(ax, violin_data,
                        #                         support, density, i,
                        #                         ["left", "right"][j])

                        # # Draw stick observations
                        # elif self.inner.startswith("stick"):
                        #     self.draw_stick_lines(ax, violin_data,
                        #                           support, density, i,
                        #                           ["left", "right"][j])

                        # Draw 68-95% lines orthogonal to violin
                        if self.inner.startswith("orth"):
                            self.draw_quartiles(ax, violin_data, support,
                                                density, max_density, i,
                                                ["left", "right"][j])

                        # The box and point interior plots are drawn for
                        # all data at the group level, so we just do that once
                        if not j:
                            continue

                        # Get the whole vector for this group level
                        violin_data = remove_na(group_data)

                        # # Draw box and whisker information
                        # if self.inner.startswith("box"):
                        #     self.draw_box_lines(ax, violin_data,
                        #                         support, density, i)

                        # # Draw point observations
                        # elif self.inner.startswith("point"):
                        #     self.draw_points(ax, violin_data, i)

                        # Draw 68-95% lines down center of violin
                        if self.inner.startswith("center"):
                            self.draw_box_lines(ax, violin_data, support,
                                                density, i, self.HDI)


                    # Option 2b: we are drawing halves of nested violins
                    # -----------------------------------------------

                    elif self.onesided:

                        # center = (i + (offsets[j] - offsets[0]))
                        if j == 0:
                            center = (i + offsets[j]/1.3)
                        else:
                            center = (i + (offsets[j]-offsets[0])/1.3)

                        # print(kws)

                        # grid = np.ones(self.gridsize) * center
                        grid = np.ones(len(plot_density)) * center
                        fill_func(plot_support,
                                  grid,
                                  grid - plot_density * self.dwidth / max_density,
                                  color=self.colors[j])

                        # ax.plot(support, grid-density * self.dwidth / max_density, color=self.gray, linewidth=self.linewidth/2)
                        ax.plot(plot_support,
                                grid-plot_density * self.dwidth / max_density,
                                color=self.gray, linewidth=self.linewidth/2,
                                zorder=2)

                        # fill_func(support,
                        #           -.01,
                        #           .01,
                        #           **kws)

                        # print(grid - density * self.dwidth)

                        # Draw the interior representation
                        if self.inner is None:
                            continue

                        # Get a nan-free vector of datapoints
                        hue_mask = self.plot_hues[i] == hue_level
                        violin_data = remove_na(group_data[hue_mask])


                        # Draw 68-95% lines down center of violin
                        if self.inner.startswith("center"):
                            self.draw_box_lines(ax, violin_data, support,
                                                density, center, self.HDI)

                        # Draw 68-95% lines orthogonal to violin
                        elif self.inner.startswith("orth"):
                            self.draw_quartiles(ax, violin_data, support,
                                                density, max_density, center)

                    # Option 2c: we are drawing full nested violins
                    # -----------------------------------------------

                    else:
                        # print(offsets[j])
                        # print(kws)
                        # print("support:", support)
                        # print("density:", density)

                        # center = (i + offsets[j]) / self.width
                        center = (i + offsets[j])

                        # print(kws)

                        # grid = np.ones(self.gridsize) * center
                        grid = np.ones(len(plot_density)) * center
                        fill_func(plot_support,
                                  grid - plot_density * self.dwidth / max_density,
                                  grid + plot_density * self.dwidth / max_density,
                                  **kws)
                        # Sometimes the first one doesn't draw the gray outline
                        fill_func(plot_support,
                                  grid - plot_density * self.dwidth / max_density,
                                  grid + plot_density * self.dwidth / max_density,
                                  **kws)
                        fill_func(plot_support,
                                  grid - plot_density * self.dwidth / max_density,
                                  grid + plot_density * self.dwidth / max_density,
                                  **kws)

                        # ax.plot(support, grid-density * self.dwidth / max_density, color=self.gray, linewidth=self.linewidth/2)
                        # ax.plot(support, grid+density * self.dwidth / max_density, color=self.gray, linewidth=self.linewidth/2)

                        # fill_func(support,
                        #           -.01,
                        #           .01,
                        #           **kws)

                        # print(grid - density * self.dwidth)

                        # Draw the interior representation
                        if self.inner is None:
                            continue

                        # Get a nan-free vector of datapoints
                        hue_mask = self.plot_hues[i] == hue_level
                        violin_data = remove_na(group_data[hue_mask])


                        # Draw 68-95% lines down center of violin
                        if self.inner.startswith("center"):
                            self.draw_box_lines(ax, violin_data, support,
                                                density, center, self.HDI)

                        # Draw 68-95% lines orthogonal to violin
                        elif self.inner.startswith("orth"):
                            self.draw_quartiles(ax, violin_data, support,
                                                density, max_density, center)

    # def draw_single_observation(self, ax, at_group, at_quant, density):
    #     """Draw a line to mark a single observation."""
    #     d_width = density * self.dwidth
    #     if self.orient == "v":
    #         ax.plot([at_group - d_width, at_group + d_width],
    #                 [at_quant, at_quant],
    #                 color=self.gray,
    #                 linewidth=self.linewidth)
    #     else:
    #         ax.plot([at_quant, at_quant],
    #                 [at_group - d_width, at_group + d_width],
    #                 color=self.gray,
    #                 linewidth=self.linewidth)

    def get_percentiles(self, support, density):
        post_f = vectorize(interp1d(support, density, bounds_error=False, fill_value=0))
        delta_x = (support[-1] - support[0])/1000
        p32 = find_dimensionless_dob_limit(
            post_f, x_mode=support[0], dob=2*.16, delta_x=delta_x, epsilon=1e-5)
        p68 = find_dimensionless_dob_limit(
            post_f, x_mode=support[0], dob=2*.84, delta_x=delta_x, epsilon=1e-5)
        p05 = find_dimensionless_dob_limit(
            post_f, x_mode=support[0], dob=2*.025, delta_x=delta_x, epsilon=1e-5)
        p95 = find_dimensionless_dob_limit(
            post_f, x_mode=support[0], dob=2*.975, delta_x=delta_x, epsilon=1e-5)
        p50 = find_dimensionless_dob_limit(
            post_f, x_mode=support[0], dob=2*.5, delta_x=delta_x, epsilon=1e-5)
        return p32, p68, p05, p95, p50

    def errfn(self, p, func, alpha, lb, ub, *args):
        def fn(x):
            pdf = func(x, *args)
            return pdf if pdf > p else 0

        # prob = integrate.quad(fn, lb, ub)[0]
        prob = trapezoid_integ_rule(fn, lb, ub, N=int(ub-lb))
        return (prob + alpha - 1.0)**2

    def get_HDI(self, support, density, alpha):
        post_f = interp1d(support, density, bounds_error=False, fill_value=0)

        def delta_f(Lambda_b, post_f, p):
            return abs(post_f(Lambda_b) - p)

        lb, ub = support[0], support[-1]
        # p is the horizontal line above which is in the HDI.
        p = fmin(self.errfn, x0=0, args=(post_f, alpha, lb, ub),
                 disp=False, ftol=1e-8)[0]
        p_array = p * np.ones(len(support)+2)
        density_array = np.array([0] + list(density) + [0])
        # Find where their difference changes sign
        idx = np.argwhere(np.diff(np.sign(density_array - p_array)) != 0).reshape(-1) + 0
        # Those points are the endpoints of the highest density interval
        try:
            Lambda_lower = support[idx[0]]
        except IndexError:
            Lambda_lower = support[0]
        try:
            Lambda_upper = support[idx[1]]
        except IndexError:
            Lambda_upper = support[-1]

        return Lambda_lower, Lambda_upper

    def draw_box_lines(self, ax, data, support, density, center, HDI):
        """Draw boxplot information at center of the density."""
        # Compute the boxplot statistics
        # q25, q50, q75 = np.percentile(data, [25, 50, 75])
        q32, q68, q05, q95, q50 = self.get_percentiles(support, density)
        if HDI:
            alpha = 1 - .68
            q32, q68 = self.get_HDI(support, density, alpha)
            alpha = 1 - .95
            q05, q95 = self.get_HDI(support, density, alpha)
        # whisker_lim = 1.5 * iqr(data)
        # whisker_lim = 1.5 * (q75 - q25)
        # h1 = np.min(data[data >= (q25 - whisker_lim)])
        # h2 = np.max(data[data <= (q75 + whisker_lim)])
        # h1, h2 = q05, q95

        # Draw a boxplot using lines and a point
        if self.orient == "v":
            ax.plot([center, center], [q05, q95],
                    linewidth=self.linewidth,
                    color=self.gray)
            ax.plot([center, center], [q32, q68],
                    linewidth=self.linewidth * 3,
                    color=self.gray)
            ax.scatter(center, q50,
                       zorder=3,
                       color="white",
                       edgecolor=self.gray,
                       s=np.square(self.linewidth * 2))
        else:
            ax.plot([q05, q95], [center, center],
                    linewidth=self.linewidth,
                    color=self.gray)
            ax.plot([q32, q68], [center, center],
                    linewidth=self.linewidth * 3,
                    color=self.gray)
            ax.scatter(q50, center,
                       zorder=3,
                       color="white",
                       edgecolor=self.gray,
                       s=np.square(self.linewidth * 2))

    def draw_quartiles(self, ax, data, support, density, max_density, center, split=False):
        """Draw the quartiles as lines at width of density."""
        # q25, q50, q75 = np.percentile(data, [25, 50, 75])
        post_f = vectorize(interp1d(support, density, bounds_error=False, fill_value=0))
        delta_x = (support[-1] - support[0])/1000
        q25 = find_dimensionless_dob_limit(
            post_f, x_mode=support[0], dob=2*.125, delta_x=delta_x)
        q75 = find_dimensionless_dob_limit(
            post_f, x_mode=support[0], dob=2*.875, delta_x=delta_x)
        q50 = find_dimensionless_dob_limit(
            post_f, x_mode=support[0], dob=2*.5, delta_x=delta_x)

        self.draw_to_density(ax, center, q25, support, density/max_density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)
        self.draw_to_density(ax, center, q50, support, density/max_density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 3] * 2)
        self.draw_to_density(ax, center, q75, support, density/max_density, split,
                             linewidth=self.linewidth,
                             dashes=[self.linewidth * 1.5] * 2)

    # def draw_points(self, ax, data, center):
    #     """Draw individual observations as points at middle of the violin."""
    #     kws = dict(s=np.square(self.linewidth * 2),
    #                color=self.gray,
    #                edgecolor=self.gray)

    #     grid = np.ones(len(data)) * center

    #     if self.orient == "v":
    #         ax.scatter(grid, data, **kws)
    #     else:
    #         ax.scatter(data, grid, **kws)

    # def draw_stick_lines(self, ax, data, support, density,
    #                      center, split=False):
    #     """Draw individual observations as sticks at width of density."""
    #     for val in data:
    #         self.draw_to_density(ax, center, val, support, density, split,
    #                              linewidth=self.linewidth * .5)

    def draw_to_density(self, ax, center, val, support, density, split, **kws):
        """Draw a line orthogonal to the value axis at width of density."""
        idx = np.argmin(np.abs(support - val))
        width = self.dwidth * density[idx] * .99

        kws["color"] = self.gray

        if self.orient == "v":
            if split == "left":
                ax.plot([center - width, center], [val, val], **kws)
            elif split == "right":
                ax.plot([center, center + width], [val, val], **kws)
            else:
                ax.plot([center - width, center + width], [val, val], **kws)
        else:
            if split == "left":
                ax.plot([val, val], [center - width, center], **kws)
            elif split == "right":
                ax.plot([val, val], [center, center + width], **kws)
            else:
                ax.plot([val, val], [center - width, center + width], **kws)

    def plot(self, ax):
        """Make the violin plot."""
        self.draw_violins(ax)
        self.annotate_axes(ax)
        if self.orient == "h":
            ax.invert_yaxis()


def violinfunctionplot(
        x=None, y=None, category=None, hue=None, data=None, order=None, hue_order=None,
        # bw="scott", cut=2,
        scale="area", scale_hue=True, gridsize=100,
        width=.8, inner="box", split=False, dodge=True, onesided=False,
        orient=None, linewidth=None, color=None, palette=None, saturation=.75, HDI=False,
        ax=None, **kwargs
        ):

    # Try to handle broken backwards-compatability
    # This should help with the lack of a smooth deprecation,
    # but won't catch everything
    warn = False
    if isinstance(x, pd.DataFrame):
        data = x
        x = None
        warn = True

    if "vals" in kwargs:
        x = kwargs.pop("vals")
        warn = True

    if "groupby" in kwargs:
        y = x
        x = kwargs.pop("groupby")
        warn = True

    if "vert" in kwargs:
        vert = kwargs.pop("vert", True)
        if not vert:
            x, y = y, x
        orient = "v" if vert else "h"
        warn = True

    msg = ("The violinplot API has been changed. Attempting to adjust your "
           "arguments for the new API (which might not work). Please update "
           "your code. See the version 0.6 release notes for more info.")
    if warn:
        warnings.warn(msg, UserWarning)

    plotter = _ViolinFunctionPlotter(
        x, y, category, hue, data, order, hue_order,
        # bw, cut,
        scale, scale_hue, gridsize,
        width, inner, split, dodge, onesided, orient, linewidth,
        color, palette, saturation, HDI)

    if ax is None:
        ax = plt.gca()

    plotter.plot(ax)
    return ax

violinfunctionplot.__doc__ = dedent("""\
    Draw a combination of boxplot and kernel density estimate.
    A violin plot plays a similar role as a box and whisker plot. It shows the
    distribution of quantitative data across several levels of one (or more)
    categorical variables such that those distributions can be compared. Unlike
    a box plot, in which all of the plot components correspond to actual
    datapoints, the violin plot features a kernel density estimation of the
    underlying distribution.
    This can be an effective and attractive way to show multiple distributions
    of data at once, but keep in mind that the estimation procedure is
    influenced by the sample size, and violins for relatively small samples
    might look misleadingly smooth.
    {main_api_narrative}
    Parameters
    ----------
    {input_params}
    {categorical_data}
    {order_vars}
    bw : {{'scott', 'silverman', float}}, optional
        Either the name of a reference rule or the scale factor to use when
        computing the kernel bandwidth. The actual kernel size will be
        determined by multiplying the scale factor by the standard deviation of
        the data within each bin.
    cut : float, optional
        Distance, in units of bandwidth size, to extend the density past the
        extreme datapoints. Set to 0 to limit the violin range within the range
        of the observed data (i.e., to have the same effect as ``trim=True`` in
        ``ggplot``.
    scale : {{"area", "count", "width"}}, optional
        The method used to scale the width of each violin. If ``area``, each
        violin will have the same area. If ``count``, the width of the violins
        will be scaled by the number of observations in that bin. If ``width``,
        each violin will have the same width.
    scale_hue : bool, optional
        When nesting violins using a ``hue`` variable, this parameter
        determines whether the scaling is computed within each level of the
        major grouping variable (``scale_hue=True``) or across all the violins
        on the plot (``scale_hue=False``).
    gridsize : int, optional
        Number of points in the discrete grid used to compute the kernel
        density estimate.
    {width}
    inner : {{"box", "quartile", "point", "stick", None}}, optional
        Representation of the datapoints in the violin interior. If ``box``,
        draw a miniature boxplot. If ``quartiles``, draw the quartiles of the
        distribution.  If ``point`` or ``stick``, show each underlying
        datapoint. Using ``None`` will draw unadorned violins.
    split : bool, optional
        When using hue nesting with a variable that takes two levels, setting
        ``split`` to True will draw half of a violin for each level. This can
        make it easier to directly compare the distributions.
    {dodge}
    {orient}
    {linewidth}
    {color}
    {palette}
    {saturation}
    {ax_in}
    Returns
    -------
    {ax_out}
    See Also
    --------
    {boxplot}
    {stripplot}
    {swarmplot}
    Examples
    --------
    Draw a single horizontal violinplot:
    .. plot::
        :context: close-figs
        >>> import seaborn as sns
        >>> sns.set_style("whitegrid")
        >>> tips = sns.load_dataset("tips")
        >>> ax = sns.violinplot(x=tips["total_bill"])
    Draw a vertical violinplot grouped by a categorical variable:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="day", y="total_bill", data=tips)
    Draw a violinplot with nested grouping by two categorical variables:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="day", y="total_bill", hue="smoker",
        ...                     data=tips, palette="muted")
    Draw split violins to compare the across the hue variable:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="day", y="total_bill", hue="smoker",
        ...                     data=tips, palette="muted", split=True)
    Control violin order by passing an explicit order:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="time", y="tip", data=tips,
        ...                     order=["Dinner", "Lunch"])
    Scale the violin width by the number of observations in each bin:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="day", y="total_bill", hue="sex",
        ...                     data=tips, palette="Set2", split=True,
        ...                     scale="count")
    Draw the quartiles as horizontal lines instead of a mini-box:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="day", y="total_bill", hue="sex",
        ...                     data=tips, palette="Set2", split=True,
        ...                     scale="count", inner="quartile")
    Show each observation with a stick inside the violin:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="day", y="total_bill", hue="sex",
        ...                     data=tips, palette="Set2", split=True,
        ...                     scale="count", inner="stick")
    Scale the density relative to the counts across all bins:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="day", y="total_bill", hue="sex",
        ...                     data=tips, palette="Set2", split=True,
        ...                     scale="count", inner="stick", scale_hue=False)
    Use a narrow bandwidth to reduce the amount of smoothing:
    .. plot::
        :context: close-figs
        >>> ax = sns.violinplot(x="day", y="total_bill", hue="sex",
        ...                     data=tips, palette="Set2", split=True,
        ...                     scale="count", inner="stick",
        ...                     scale_hue=False, bw=.2)
    Use ``hue`` without changing violin position or width:
    .. plot::
        :context: close-figs
        >>> tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
        >>> ax = sns.violinplot(x="day", y="total_bill", hue="weekend",
        ...                     data=tips, dodge=False)
    Draw horizontal violins:
    .. plot::
        :context: close-figs
        >>> planets = sns.load_dataset("planets")
        >>> ax = sns.violinplot(x="orbital_period", y="method",
        ...                     data=planets[planets.orbital_period < 1000],
        ...                     scale="width", palette="Set3")
    Draw a violin plot on to a :class:`FacetGrid` to group within an additional
    categorical variable:
    .. plot::
        :context: close-figs
        >>> g = sns.FacetGrid(tips, col="time", size=4, aspect=.7)
        >>> (g.map(sns.violinplot, "sex", "total_bill", "smoker", split=True)
        ...   .despine(left=True)
        ...   .add_legend(title="smoker"))  # doctest: +ELLIPSIS
        <seaborn.axisgrid.FacetGrid object at 0x...>
    """)#.format(**_categorical_docs)
