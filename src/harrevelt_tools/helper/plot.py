import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import itertools


class ListPlot:
    def __init__(self, image_list, **kwargs):
        """
        Here you can insert stuff
        
        :param image_list:
        :param kwargs:
        """
        figsize = kwargs.get('figsize', (10, 10))
        fignum = kwargs.get('fignum')
        dpi = kwargs.get('dpi')

        title_string = kwargs.get('title', "")
        self.sub_title = kwargs.get('subtitle', None)
        self.cbar_ind = kwargs.get('cbar', False)
        self.cbar_round_n = kwargs.get('cbar_round_n', 2)
        self.cmap = kwargs.get('cmap', 'gray')

        self.vmin = kwargs.get('vmin', None)
        self.ax_off = kwargs.get('ax_off', False)
        self.augm_ind = kwargs.get('augm', None)
        self.aspect_mode = kwargs.get('aspect', 'equal')
        self.start_square_level = kwargs.get('start_square_level', 8)
        self.sub_col_row = kwargs.get('sub_col_row', None)

        self.wspace = kwargs.get('wspace', 0.1)
        self.hspace = kwargs.get('hspace', 0.1)

        self.debug = kwargs.get('debug', False)

        self.figure = plt.figure(fignum, figsize=figsize, dpi=dpi)
        self.figure.suptitle(title_string)
        self.canvas = self.figure.canvas
        # Used to go from positive to negative scaling
        self.epsilon = 0.001

        # Only when we have an numpy array
        if isinstance(image_list, np.ndarray):
            # With just two dimensions..
            if image_list.ndim == 2:
                # Add one..
                image_list = image_list[np.newaxis]

        n_rows = len(image_list)
        self.gs0 = gridspec.GridSpec(n_rows, 1, figure=self.figure)
        self.gs0.update(wspace=self.wspace, hspace=self.hspace)  # set the spacing between axes.
        self.gs0.update(top=1. - 0.5 / (n_rows + 1), bottom=0.5 / (n_rows + 1))
        # left = 0.5 / (ncol + 1), right = 1 - 0.5 / (ncol + 1))

        if self.debug:
            print("Status of loaded array")
            print("\tNumber of rows//length of image list ", n_rows)
            if hasattr(image_list, 'ndim'):
                print("\tDimension of image list", image_list.ndim)
            if hasattr(image_list[0], 'ndim'):
                print("\tDimension of first image list element ", image_list[0].ndim)

        self.ax_list, self.ax_imshow_list, self.ax_cbar_list = self.plot_3d_list(image_list)
        self.ax_list = list(itertools.chain(*self.ax_list))
        self.ax_imshow_list = list(itertools.chain(*self.ax_imshow_list))
        self.ax_cbar_list = list(itertools.chain(*self.ax_cbar_list))

        self.press_indicator = False
        self.press_position = None
        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('motion_notify_event', self.move_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)

    def button_press_callback(self, input):
        if input.inaxes in self.ax_list:
            index_current_ax = self.ax_list.index(input.inaxes)
            self.temp_ax = self.ax_imshow_list[index_current_ax]

            # Reset colorbar
            if input.button == MouseButton.RIGHT:
                temp_array = self.temp_ax.get_array()
                temp_cbar = self.ax_cbar_list[index_current_ax]
                reset_clim = [temp_array.min(), temp_array.max()]
                self.temp_ax.set_clim(reset_clim)
                if temp_cbar:
                    temp_cbar.set_ticks(reset_clim)
                self.canvas.draw()
            elif input.button == MouseButton.LEFT:
                self.press_indicator = True
                self.press_position = {'x': input.xdata, 'y': input.ydata}
                self.press_clim = list(self.temp_ax.get_clim())
            else:
                print('Unknown button pressed ', input.button, input)

            if self.debug:
                print("You have clicked", input, self.ax_list.index(input.inaxes))
                print("Previous clim ", self.press_clim)

    def move_callback(self, input):
        if input.inaxes in self.ax_list:
            if self.press_indicator:
                current_position = {'x': input.xdata, 'y': input.ydata}
                index_current_ax = self.ax_list.index(input.inaxes)

                size_x, size_y = self.temp_ax.get_size()
                distance_x = (current_position['x'] - self.press_position['x']) / size_x
                distance_y = (current_position['y'] - self.press_position['y']) / size_y

                if self.press_clim[1] < 0:
                    distance_x = 1 - distance_x
                else:
                    distance_x = 1 + distance_x

                if self.press_clim[0] < 0:
                    distance_y = 1 - distance_y
                else:
                    distance_y = 1 + distance_y

                if np.abs(self.press_clim[1]) < self.epsilon:
                    if self.press_clim[1] > 0:
                        self.press_clim[1] = -2 * self.epsilon
                    else:
                        self.press_clim[1] = 2 * self.epsilon

                if np.abs(self.press_clim[0]) < self.epsilon:
                    if self.press_clim[0] > 0:
                        self.press_clim[0] = - 2 * self.epsilon
                    else:
                        self.press_clim[0] = 2 * self.epsilon

                max_clim = self.press_clim[1] * distance_x
                min_clim = self.press_clim[0] * distance_y

                new_clim = [min_clim, max_clim]
                self.temp_ax.set_clim(new_clim)

                if self.cbar_ind:
                    temp_cbar = self.ax_cbar_list[index_current_ax]
                    temp_cbar.set_ticks(new_clim)
                self.canvas.draw()

    def button_release_callback(self, input):
        # if input.inaxes:
        self.press_indicator = False
        self.press_position = None

    def plot_3d_list(self, image_list):
        # Input of either a 2d list of np.arrays.. or a 3d list of np.arrays..
        ax_list = []
        ax_imshow_list = []
        ax_cbar_list = []
        for i, i_gs in enumerate(self.gs0):
            sub_ax_list = []
            sub_ax_imshow_list = []
            sub_cbar_list = []
            temp_img = image_list[i]
            # Below we can configure how exactly we want to order the images....
            if self.sub_col_row is not None:
                n_sub_col, n_sub_row = self.sub_col_row
            else:
                # Here we split between having a numpy array
                # or a list...
                if hasattr(temp_img, 'ndim') and hasattr(temp_img, 'shape') and hasattr(temp_img, 'reshape'):
                    if temp_img.ndim == 4:
                        n_sub_col = temp_img.shape[0]
                        n_sub_row = temp_img.shape[1]
                        # With this we want to prevent plotting a 3D array in the next step
                        temp_img = temp_img.reshape((n_sub_col * n_sub_row,) + temp_img.shape[2:])
                    elif temp_img.ndim == 3:
                        n_sub_col = temp_img.shape[0]
                        if n_sub_col > self.start_square_level:
                            n_sub_col = n_sub_row = n_sub_col // 2
                            # n_sub_col, n_sub_row = hmisc.get_square(n_sub_col)
                            print('Using sub col, sub row:', n_sub_col, n_sub_row)
                        else:
                            n_sub_row = 1
                    else:
                        temp_img = temp_img[np.newaxis]
                        n_sub_col = 1
                        n_sub_row = 1
                    if self.debug:
                        print("\tTemp image is nparray and has shape ", temp_img.shape)
                        print("\tUsed col/row ", n_sub_col, n_sub_row)
                else:
                    n_sub_col = len(temp_img)
                    n_sub_row = 1
                    if self.debug:
                        print("\tTemp image is list and has length ", len(temp_img))
                        print("\tUsed col/row ", n_sub_col, n_sub_row)
            # If we want to specifcy the vmin per list item.. we can do that here..
            if isinstance(self.vmin, list):
                sel_vmin = self.vmin[i]
            else:
                sel_vmin = self.vmin

            for j, ii_gs in enumerate(i_gs.subgridspec(n_sub_row, n_sub_col, wspace=self.wspace, hspace=self.hspace)):
                # Do not continue plotting when we are exceeding the number of things to plot
                # This avoids trying to plot stuff in an axes when everything is already plotted.
                if j >= len(temp_img):
                    print("STOP")
                    break
                # Hacky way to fix the list in list vmin specification
                if isinstance(sel_vmin, list):
                    sel_sel_vmin = sel_vmin[j]
                else:
                    sel_sel_vmin = sel_vmin

                ax = self.figure.add_subplot(ii_gs)
                if self.augm_ind:
                    plot_img = eval('{fun}({var})'.format(fun=self.augm_ind, var=str('temp_img[j]')))
                    if 'angle' in self.augm_ind:
                        sel_sel_vmin = (-np.pi, np.pi)
                else:
                    plot_img = temp_img[j]

                if self.debug:
                    print(f'shape {i} - {len(temp_img)}', end=' \t|\t')
                    print(f'row/col {n_sub_row} - {n_sub_col}', end=' \t|\t')
                    print(f'shape {j} - {plot_img.shape}', end=' \t|\n')

                if self.cmap == 'rgb':
                    # For this to work.. there are some prequisites..
                    # First: shape of array should be: (1, nx, ny, 3)
                    # Second: Array should be given like [plot_array]
                    # Third: Give a sub_col_row with a value
                    # Last: of course, set cmap=rgb
                    map_ax = ax.imshow(plot_img, vmin=sel_sel_vmin, aspect=self.aspect_mode)
                else:
                    map_ax = ax.imshow(plot_img, vmin=sel_sel_vmin, aspect=self.aspect_mode, cmap=self.cmap)

                if self.cbar_ind:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    temp_cbar = plt.colorbar(map_ax, cax=cax)
                    if sel_sel_vmin is None:
                        vmin_temp = [plot_img.min(), plot_img.max()]
                        vmin_temp = list(map(float, vmin_temp))
                        map_ax.set_clim(vmin_temp)
                        temp_cbar.set_ticks([np.round(x, self.cbar_round_n) for x in vmin_temp])
                    else:
                        map_ax.set_clim(sel_sel_vmin)
                        temp_cbar.set_ticks([np.round(x, self.cbar_round_n) for x in sel_sel_vmin])
                else:
                    temp_cbar = None

                if self.sub_title is not None:
                    ax.set_title(self.sub_title[i][j])
                if self.ax_off:
                    ax.set_axis_off()

                sub_ax_list.append(ax)
                sub_ax_imshow_list.append(map_ax)
                sub_cbar_list.append(temp_cbar)

            ax_list.append(sub_ax_list)
            ax_imshow_list.append(sub_ax_imshow_list)
            ax_cbar_list.append(sub_cbar_list)

        return ax_list, ax_imshow_list, ax_cbar_list
