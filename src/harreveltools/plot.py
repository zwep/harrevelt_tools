import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import Tuple, List, Union
import os
import numpy as np
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import itertools
from .data_transform import scale_minmax, get_proper_scaled_v2, get_square


class ListPlot:
    # Used in scaling when dragging it..
    epsilon = 0.001

    def __init__(self, image_list,
                 col_row: Tuple[int, int] = None,
                 ax_off: bool = False,
                 **kwargs):
        # What can image_list be...
        # list of 2D arrays (with different sizes) --> essentially a 3D array
        # OR one 3D numpy array --> do nothing?
        # OR one 2D numpy array (after adding one axis)
            # dit zou dus standaard naar (1, len()) gaan
            # Maar soms wilde ik ook (len(), 1) bijvoorbeeld
        # Dus het komt er op neer dat ik 4D arrays plot
        # 4D in de vorm van
        # [X, Y, Z] --> [X[None], Y[None], Z[None]]
        # [[X], [Y], [Z] --> [X[None], Y[None], Z[None]]
        # X -> [X[None]]
        # Het voordeel met een lijst is dus dat ik verschillende vormen en maten tegelijk kan plotten

        # Ik moet dus alles wat analoog is aan een >4D array niet accepteren.
        # Of die moet ik veranderen naar 4D raveled arrays...
        image_list = self._update_image_list(image_list)
        """ Titles, cbar, axis options """
        self.title_string = kwargs.get('title', "")
        # TODO: only check is now that sub_title string is not None.
        # If that is the case.. then it enters an element (which is not possible when string is given)
        self.sub_title_string = kwargs.get('subtitle', None)  # Can be None, string,
        self.cbar_bool = kwargs.get('cbar', False)
        self.cbar_float_rounding_n = kwargs.get('cbar_round_n', 2)
        self.cmap = kwargs.get('cmap', 'gray')
        # This can also be a list of lists..
        self.vmin = kwargs.get('vmin', None)
        self.ax_off = ax_off
        self.aspect_mode = kwargs.get('aspect', 'equal')

        """ Create figure and offsets """
        figsize = kwargs.get('figsize', (10, 10))
        fignum = kwargs.get('fignum')
        dpi = kwargs.get('dpi', 300)
        # The w- and h-space is  the distance between the main grid-spec items
        self.wspace = kwargs.get('wspace', 0.1)
        self.hspace = kwargs.get('hspace', 0.1)

        self.figure = plt.figure(fignum, figsize=figsize, dpi=dpi)
        self.figure.suptitle(self.title_string)
        self.canvas = self.figure.canvas

        """
        Advanced editing functions
        augm - enables to use a function toa ugment each individual list item
        proper_scaling sets the correct scaling (estimated by a function) for each plotted image 
        """
        # These apply advanced functions to the input data...
        self.augm_ind = kwargs.get('augm', None)
        self.proper_scaling = kwargs.get('proper_scaling', False)

        # This should be done better...
        if self.proper_scaling is True:
            if self.vmin is None:
                # TODO: This does not work when a list of list is given..
                if isinstance(image_list, list):
                    self.vmin = []
                    temp_img_list = []
                    for i_array in image_list:
                        temp_vmin = [(0, get_proper_scaled_v2(x, patch_shape=128, stride=32)) for x in i_array]
                        temp_img = scale_minmax(i_array, axis=(-2, -1))
                        self.vmin.append(temp_vmin)
                        temp_img_list.append(temp_img)
                    image_list = temp_img_list
                else:
                    self.vmin = [(0, get_proper_scaled_v2(x, patch_shape=128, stride=32)) for x in image_list]
                    image_list = scale_minmax(image_list, axis=(-2, -1))
            else:
                print('Vmin is set AND proper scaling is turned on. Defaulting to vmin')

        self.debug = kwargs.get('debug', False)



        """
        Here we create the major grid-spec item.
        This will be divided into n_row and n_col. Normally we will have 1 row and X amount of cols. 
        Where X is the length of the image_list.
        
        Later, we will 
        """

        """ col (row) / sub col (row) options """
        # TODO: when to use col row when to use sub col row?
        # TODO: this is not fair to use len image_list
        n_col, n_row = (1, len(image_list)) if col_row is None else col_row
        # square_level is used to split sub_col_row objects
        self.start_square_level = kwargs.get('start_square_level', 8)
        self.sub_col_row = kwargs.get('sub_col_row', None)

        self.gridspec_row_col = gridspec.GridSpec(n_row, n_col, figure=self.figure)
        self.gridspec_row_col.update(wspace=self.wspace, hspace=self.hspace)  # set the spacing between axes.
        self.gridspec_row_col.update(top=1. - 0.5 / (n_row + 1), bottom=0.5 / (n_row + 1))
        # left = 0.5 / (ncol + 1), right = 1 - 0.5 / (ncol + 1))

        self.ax_list, self.ax_imshow_list, self.ax_cbar_list = self.plot_3d_list(image_list)
        self.ax_list = list(itertools.chain(*self.ax_list))
        self.ax_imshow_list = list(itertools.chain(*self.ax_imshow_list))
        self.ax_cbar_list = list(itertools.chain(*self.ax_cbar_list))

        """
        This allows for scaling of the images by using Left click (mouse) + drag
        Resets the scaling with Right click (mouse)
        """
        self.press_indicator = False
        self.press_position = None

        self.canvas.mpl_connect('button_press_event', self.button_press_callback)
        self.canvas.mpl_connect('motion_notify_event', self.move_callback)
        self.canvas.mpl_connect('button_release_event', self.button_release_callback)

    @staticmethod
    def _update_ndarray_dimensions(x, add_dimension=True):
        if x.ndim < 1:
            print()  # TODO too few dimensions
            return
        elif x.ndim == 2:
            x = x[np.newaxis]  # Convert it to a list with a 3D array
            if add_dimension:
                x = [x]
        elif x.ndim == 3:
            if add_dimension:
                x = [x]  # Convert it to a list with a 3D array
        elif x.ndim > 4:
            # Reshape anything above four dimensions to 4D
            if add_dimension:
                x = x.reshape((-1,) + x.shape[-3:])
            else:
                x = x.reshape((-1,) + x.shape[-2:])
        return x

    def _update_image_list(self, image_list):
            # Do more fancy operations to align the content of image_list
            # One thing is:
            # Check if it is a list
            if isinstance(image_list, list):
                # Check if each element is an nd.array
                if all([isinstance(x, np.ndarray) for x in image_list]):
                    # Each element is atleast an ndarray.
                    # Correct the dimensions so that we have a list of 3D objects
                    image_list = [self._update_ndarray_dimensions(x, add_dimension=False) for x in image_list]
                else:
                    # Not evrything is an ndarray. That is fine
                    if all([isinstance(x, list) or isinstance(x, np.ndarray) for x in image_list]):
                        # Okay, so we have a nested list and mixed list. That is fine. I guess
                        # As long as the content of each nested list is an ndarray
                        # Shouldnt those arrays be of dimension two *questionmark*
                        check_content = []
                        for ii_item, _ in enumerate(image_list):
                            # This i_item is a list or an ndarray
                            if isinstance(image_list[ii_item], np.ndarray):
                                # We have this situation [X],
                                # We can correct a little bit
                                image_list[ii_item] = self._update_ndarray_dimensions(image_list[ii_item], add_dimension=False)
                            else:
                                # Now we have a nested list...
                                for ix in image_list[ii_item]:
                                    # Now it should be a 2D array
                                    if isinstance(ix, np.ndarray):
                                        if ix.ndim == 2:
                                            check_content.append(True)
                                        else:
                                            check_content.append(False)
                                    # If it is not a nd array.. then forget it
                                    else:
                                        check_content.append(False)
                        # If this is all goood, even with some minor corrections, then good... duh
                        if all(check_content):
                            pass
                        else:
                            print()  # TODO: print list violation.
                            return
                    else:
                        print()  # TODO not everything is plottable it seems
                        return
            else:
                if isinstance(image_list, np.ndarray):
                    # We simply have an ndarray, correct for dimensions
                    image_list = self._update_ndarray_dimensions(image_list)
                else:
                    print()  # TODO it is not a list.. and not an ndarray
                    return
            return image_list

    def savefig(self, name, home=True, pad_inches=0.0, bbox_inches='tight'):
        if name.endswith(".png"):
            name = name[:-4]

        if home:
            dest_path = os.path.expanduser(f"~/{name}.png")
        else:
            dest_path = name

        print(f'Saving image to {dest_path}')
        self.figure.savefig(dest_path, bbox_inches=bbox_inches, pad_inches=pad_inches)

    def button_press_callback(self, input):
        """
        Used to allow for color scaling with the left mouse button
        Reset color scaling with right mouse button
        :param input:
        :return:
        """
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
        """
        Allows for dragging and tracking of the mouse movement in x- and y-direction
        :param input:
        :return:
        """
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

                if self.cbar_bool:
                    temp_cbar = self.ax_cbar_list[index_current_ax]
                    temp_cbar.set_ticks(new_clim)
                self.canvas.draw()

    def button_release_callback(self, input):
        """
        Resets the press indication and the position
        :param input:
        :return:
        """
        self.press_indicator = False
        self.press_position = None

    def plot_3d_list(self, image_list):
        """
        Here we execute the plotting of the image_list
        :param image_list:
        :return:
        """
        # Input of either a 2d list of np.arrays.. or a 3d list of np.arrays..
        ax_list = []
        ax_imshow_list = []
        ax_cbar_list = []

        for i, i_gs in enumerate(self.gridspec_row_col):
            # TODO Why is this
            if i >= len(image_list):
                print(f"STOP - length is violated {i} >= {len(image_list)}")
                break
            sub_ax_list = []
            sub_ax_imshow_list = []
            sub_cbar_list = []
            temp_img = image_list[i]
            # TODO this should be done A LOT better
            # Here we split between having a numpy array
            # TODO oh boy.. the sub col row
            # or a list...
            if hasattr(temp_img, 'ndim') and hasattr(temp_img, 'shape') and hasattr(temp_img, 'reshape'):
                if temp_img.ndim == 4:
                    n_sub_col = temp_img.shape[0]
                    n_sub_row = temp_img.shape[1]
                    # With this we want to prevent plotting a 3D array in the next step
                    # But avoid this with rgb images..
                    if self.cmap == 'rgb':
                        # Make atleast check something I guess
                        assert temp_img.shape[-1] == 3, "Last image dimension is not equal to three"
                    else:
                        #
                        temp_img = temp_img.reshape((n_sub_col * n_sub_row,) + temp_img.shape[2:])
                elif temp_img.ndim == 3:
                    n_sub_col = temp_img.shape[0]
                    if n_sub_col > self.start_square_level:
                        n_sub_col, n_sub_row = get_square(n_sub_col)
                        print('Using sub col, sub row:', n_sub_col, n_sub_row)
                    else:
                        n_sub_row = 1
                elif temp_img.ndim == 2:
                    temp_img = temp_img[np.newaxis]
                    n_sub_col = 1
                    n_sub_row = 1
                else:
                    print('Unknown image dimension: ', temp_img.shape)
                    return
            else:
                n_sub_col = len(temp_img)
                n_sub_row = 1

            # If we have configured sub col row.. we want to impose that now
            if self.sub_col_row is not None:
                # If this is selected.. then we expect a list of 3D arrays...
                # Is that desireable?
                n_sub_col, n_sub_row = self.sub_col_row

            if self.debug:
                print(f"\tVariable temp_image is has length {len(temp_img)} and shape {temp_img.shape}")
                print("\tThe number of col/row is set to", n_sub_col, n_sub_row)

            # If we want to specifcy the vmin per list item.. we can do that here..
            if isinstance(self.vmin, list):
                sel_vmin = self.vmin[i]
            else:
                sel_vmin = self.vmin

            for j, ii_gs in enumerate(i_gs.subgridspec(n_sub_row, n_sub_col, wspace=self.wspace, hspace=self.hspace)):
                # Do not continue plotting when we are exceeding the number of things to plot
                # This avoids trying to plot stuff in an axes when everything is already plotted.
                if j >= len(temp_img):
                    print(f"STOP - length is violated {j} >= {len(temp_img)}")
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
                    if np.iscomplexobj(temp_img[j]):
                        # If we did not choose anything, default to 'abs'
                        plot_img = np.abs(temp_img[j])
                    else:
                        plot_img = temp_img[j]

                if self.debug:
                    print(f'Image id: {i} - shape of temp image {temp_img.shape}', end=' \t|\n')
                    print(f'Plot id: {j} - shape of plot image {plot_img.shape}', end=' \t|\n')

                if self.cmap == 'rgb':
                    # For this to work.. there are some prequisites..
                    # First: shape of array should be: (1, nx, ny, 3)
                    # Second: Array should be given like [plot_array]
                    # Third: Give a sub_col_row with a value
                    # Last: of course, set cmap=rgb
                    map_ax = ax.imshow(plot_img, vmin=sel_sel_vmin, aspect=self.aspect_mode)
                elif isinstance(self.cmap, list):
                    map_ax = ax.imshow(plot_img, vmin=sel_sel_vmin, aspect=self.aspect_mode, cmap=self.cmap[i][j])
                else:
                    map_ax = ax.imshow(plot_img, vmin=sel_sel_vmin, aspect=self.aspect_mode, cmap=self.cmap)

                if self.cbar_bool:
                    divider = make_axes_locatable(ax)
                    cax = divider.append_axes("right", size="5%", pad=0.05)
                    temp_cbar = plt.colorbar(map_ax, cax=cax)
                    if sel_sel_vmin is None:
                        vmin_temp = [plot_img.min(), plot_img.max()]
                        vmin_temp = list(map(float, vmin_temp))
                        map_ax.set_clim(vmin_temp)
                        temp_cbar.set_ticks([np.round(x, self.cbar_float_rounding_n) for x in vmin_temp])
                    else:
                        map_ax.set_clim(sel_sel_vmin)
                        # temp_cbar.set_ticks([np.round(x, self.cbar_float_rounding_n) for x in sel_sel_vmin])
                else:
                    temp_cbar = None

                if self.sub_title_string is not None:
                    ax.set_title(self.sub_title_string[i][j])
                if self.ax_off:
                    ax.set_axis_off()

                sub_ax_list.append(ax)
                sub_ax_imshow_list.append(map_ax)
                sub_cbar_list.append(temp_cbar)

            ax_list.append(sub_ax_list)
            ax_imshow_list.append(sub_ax_imshow_list)
            ax_cbar_list.append(sub_cbar_list)

        return ax_list, ax_imshow_list, ax_cbar_list


