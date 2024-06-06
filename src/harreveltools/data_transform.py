import pandas as pd
import collections
import itertools
import os

import numpy as np
import pydicom
from PIL import Image

import warnings
import skimage

import cv2
import scipy.ndimage.interpolation as scint

from scipy.spatial import ConvexHull
import scipy.signal
import scipy.ndimage
import skimage.transform as sktransf
from PIL import Image, ImageDraw
import numbers
from typing import Tuple, List, Union
import scipy.optimize
import skimage.util.dtype
import skimage.transform as sktransform
import math

"""

There is no clear concensus in my head what 'analytics'  should be and what 'data_transform'  should be


"""


def listdict2dictlist(list_dict):
    # Transforms a list of dictionaries to a dictionary of lists
    set_of_keys = set(itertools.chain(*[list(x.keys()) for x in list_dict]))
    temp_dict = {k: [] for k in set_of_keys}
    for i_key in set_of_keys:
        temp_list = [[v for k, v in x_dict.items() if k == i_key] for x_dict in list_dict]
        temp_list = list(itertools.chain(*temp_list))
        temp_dict[i_key].extend(temp_list)

    return temp_dict


def dictlist2listdict(dict_list):
    # Transforms a dictionary of lists to a list of dictionaries
    return [dict(zip(dict_list, t)) for t in zip(*dict_list.values())]


def aggregate_dict_mean_value(d, level=0, agg_dict=None):
    if agg_dict is None:
        agg_dict = {}
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            aggregate_dict_mean_value(v, level + 1)
        else:
            # Unfold/unravel anything that is nested, and thus not a dict..
            # Not sure if this is truely desired...
            unraveled = v
            while isinstance(unraveled[0], collections.Iterable):
                unraveled = list(itertools.chain(*unraveled))
            unraveled = np.array(unraveled)
            unraveled[np.isinf(unraveled)] = 0
            agg_dict.update({k: np.mean(unraveled)})
    return agg_dict


def aggregate_dict_std_value(d, level=0, agg_dict=None):
    # A simple copy because I think that is more insightful
    if agg_dict is None:
        agg_dict = {}
    # Prints a dictionary
    for k, v in d.items():
        if isinstance(v, dict):
            aggregate_dict_std_value(v, level + 1)
        else:
            # Unfold/unravel anything that is nested, and thus not a dict..
            # Not sure if this is truely desired...
            unraveled = v
            while isinstance(unraveled[0], collections.Iterable):
                unraveled = list(itertools.chain(*unraveled))
            unraveled = np.array(unraveled)
            unraveled[np.isinf(unraveled)] = 0
            agg_dict.update({k: np.std(unraveled)})
    return agg_dict


def create_random_center_mask(x_shape, random=False, mask_fraction=0.07):
    # Used for optimizing the shim region and as reference for the signal intensity to scale the b1p with
    n_y, n_x = x_shape[-2:]
    y_offset = 0
    x_offset = 0
    if random:
        y_offset = np.random.randint(-n_y // 12, n_y // 12)
        x_offset = np.random.randint(-n_x // 12, n_x // 12)
    y_center = n_y // 2 + y_offset
    x_center = n_x // 2 + x_offset
    center_mask = np.zeros((n_y, n_x))
    delta_x = int(mask_fraction * n_x)
    delta_y = int(mask_fraction * n_y)
    center_mask[y_center - delta_y:y_center + delta_y, x_center - delta_x:x_center + delta_x] = 1
    return center_mask


def convert_cpx2int8(x, stack_axis=1):
    x_int8_stacked = np.stack([skimage.util.dtype._convert(x.real, np.int8),
                               skimage.util.dtype._convert(x.imag, np.int8)],
                               axis=stack_axis)
    return x_int8_stacked


def convert_cpx2int16(x, stack_axis=1):
    x_int8_stacked = np.stack([skimage.util.dtype._convert(x.real, np.int16),
                               skimage.util.dtype._convert(x.imag, np.int16)],
                               axis=stack_axis)
    return x_int8_stacked


def running_mean(x, N):
    return np.convolve(x, np.ones(N) / N, mode='same')


def to_shape(x, target_shape):
    padding_list = []
    for x_dim, target_dim in zip(x.shape, target_shape):
        pad_value = (target_dim - x_dim)
        pad_tuple = ((pad_value//2, pad_value//2 + pad_value%2))
        padding_list.append(pad_tuple)

    return np.pad(x, tuple(padding_list), mode='constant')


def get_smooth_kernel(n):
    smooth_kernel = np.ones((n, n)) / n ** 2
    return smooth_kernel


def get_center_transformation_coords(img) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """
    Used to do an affine transformation on a mask to shift it to the middle.
    Also returns the shift coordinates
    :param img:
    :return:
    """
    mid_x_shape, mid_y_shape = np.array(img.shape) // 2

    sum_ax0 = np.sum(img, axis=0)
    sum_ax1 = np.sum(img, axis=1)
    # Find middle of the mask.....
    # With this we want to register the stuff..
    index_ax_0 = np.argwhere((sum_ax0 > 0))
    index_ax_1 = np.argwhere((sum_ax1 > 0))
    if len(index_ax_0):
        min_x = int(np.min(index_ax_0))
        max_x = int(np.max(index_ax_0))
    else:
        min_x = 0
        max_x = img.shape[0]

    if len(index_ax_1):
        min_y = int(np.min(index_ax_1))
        max_y = int(np.max(index_ax_1))
    else:
        min_y = 0
        max_y = img.shape[1]

    mid_x = min_x + (max_x - min_x) / 2
    mid_y = min_y + (max_y - min_y) / 2
    affine_x = int(mid_x_shape - mid_x)
    affine_y = int(mid_y_shape - mid_y)

    crop_coords = (min_x, max_x, min_y, max_y)
    affine_coords = (affine_x, affine_y)

    return affine_coords, crop_coords


def apply_center_transformation(x, affine_coords: Tuple[int, ...], crop_coords: Tuple[int, ...], dtype=None):
    # Affine transform is used as off set for the x- and y-coords
    # The crop coords are used to select only that part of the image which are relevant (xmin, xmax, ymin, ymax)
    # If dtype is None then it defaults to np.float64
    x_affine = np.zeros(x.shape, dtype=dtype)
    min_x, max_x, min_y, max_y = crop_coords
    affine_x, affine_y = affine_coords

    # The code below can definitely be made easier/shorter
    # But this explicit form works and that is fine for now :)
    # The reasoning behind is that we want to allow for an affine transformation OVER the image boundaries
    min_x_shifted = min_x + affine_x
    delta_min_x = 0
    if min_x_shifted < 0:
        delta_min_x = abs(min_x_shifted)
        min_x_shifted = 0

    max_x_shifted = max_x + affine_x
    delta_max_x = -1
    if max_x_shifted >= x.shape[1]:
        delta_max_x = max_x_shifted - x.shape[1]
        max_x_shifted = x.shape[1]

    min_y_shifted = min_y + affine_y
    delta_min_y = 0
    if min_y_shifted < 0:
        delta_min_y = abs(min_y_shifted)
        min_y_shifted = 0

    max_y_shifted = max_y + affine_y
    delta_max_y = -1
    if max_y_shifted >= x.shape[0]:
        delta_max_y = max_y_shifted - x.shape[0]
        max_y_shifted = x.shape[0]

    # We need the +1 otherwise we cut off part of the figure..
    cropped_x = x[min_y + delta_min_y:max_y - delta_max_y, min_x + delta_min_x:max_x - delta_max_x]
    x_affine[min_y_shifted: max_y_shifted+1, min_x_shifted: max_x_shifted+1] = cropped_x
    return x_affine


def apply_crop(x, crop_coords, marge=0):
    # Possibility to add a marge on the width of the crop.. Might prove to be useful...
    # Usable only for 2D images..
    # Order is (y, x)
    img_shape = x.shape
    min_x, max_x, min_y, max_y = crop_coords
    min_x = max(0, min_x)
    max_x = min(img_shape[1], max_x)

    min_y = max(0, min_y)
    max_y = min(img_shape[0], max_y)

    cropped_x = x[(min_y-marge):(max_y+marge), (min_x-marge):(max_x+marge)]
    return cropped_x


def apply_crop_axis(x, crop_coords, marge=0, axis=0):
    # Possibility to add a marge on the width of the crop.. Might prove to be useful...
    # Usable only for 2D images..
    # Order is (y, x)
    img_shape = x.shape
    min_x, max_x, min_y, max_y = crop_coords
    min_x = max(0, min_x)
    max_x = min(img_shape[1], max_x)

    min_y = max(0, min_y)
    max_y = min(img_shape[0], max_y)

    if axis == 0:
        cropped_x = x[(min_y-marge):(max_y+marge), :]
    else:
        cropped_x = x[:, (min_x - marge):(max_x + marge)]
    return cropped_x


def get_square(x):
    """
    Used to get an approximation of the square of a number.
    Needed to place N plots on a equal sized grid.
    :param x:
    :return:
    """
    x_div = simple_div(x)
    x_sqrt = math.sqrt(x)
    diff_list = [abs(y - x_sqrt) for y in x_div]
    res = diff_list.index(min(diff_list))
    # Return largest first...
    return sorted([x_div[res], x // x_div[res]])[::-1]


def get_crop_coords_center(image_shape, width):
    # Assuming 2D image
    # Cuts out a square with the given width from the center
    # Compatible with apply_crop
    # min_x, max_x, min_y, max_y = crop_coords
    n_y = image_shape[0]
    n_x = image_shape[1]
    mid_y = n_y//2
    mid_x = n_x//2

    min_x = mid_x - width//2
    max_x = mid_x + width // 2

    min_y = mid_y - width // 2
    max_y = mid_y + width // 2

    min_x = max(0, min_x)
    max_x = min(image_shape[1], max_x)

    min_y = max(0, min_y)
    max_y = min(image_shape[0], max_y)

    return min_x, max_x, min_y, max_y


def get_crop(x, x_mask=None, silent=False, accept_treshold=0.01):
    # Simple..yet effective..?
    if x_mask is None:
        if not silent:
            print('It is best to supply a mask...')
            print('I could create a function.. but they are not robust')
        x_mask = get_treshold_label_mask(x)

    # This checks if the number of masked elements is larger than
    # a percentage of the amount of pixels larger than the mean
    if np.sum(x_mask) > accept_treshold * np.sum((x > np.mean(x))):
        # We need the mask to determine the affine transfomration to put it in the middle..
        _, crop_coords = get_center_transformation_coords(x_mask)
        x = apply_crop(x, crop_coords=crop_coords)
        x_mask = apply_crop(x_mask, crop_coords=crop_coords)
    else:
        x_mask = np.ones(x.shape)

    return x, x_mask


def get_center_transformation(x, x_mask=None):
    if x_mask is None:
        print('It is best to supply a mask...')
        print('I could create a function.. but they are not robust')
        x_mask = get_treshold_label_mask(x)

    # We need the mask to determine the affine transfomration to put it in the middle..
    affine_coords, crop_coords = get_center_transformation_coords(x_mask)
    x = apply_center_transformation(x, affine_coords=affine_coords, crop_coords=crop_coords)
    x_mask = apply_center_transformation(x_mask, affine_coords=affine_coords, crop_coords=crop_coords)
    return x, x_mask


def get_center_transformation_3d(x, x_mask=None):
    # Now performs the shifting over ALL the slices.
    if x_mask is None:
        x_mask = np.array([get_treshold_label_mask(y) for y in x])

    # Returns shifted X and associated mask
    # Assumes input of (slices, x, y)
    max_slice = x.shape[0]

    x_shifted = []
    x_mask_shifted = []
    for i_slice in range(max_slice):
        mask_slice = x_mask[i_slice]
        temp_affine_coords, temp_crop_coords = get_center_transformation_coords(mask_slice)
        temp_x_shifted = apply_center_transformation(x[i_slice], affine_coords=temp_affine_coords, crop_coords=temp_crop_coords)
        temp_x_mask_shifted = apply_center_transformation(mask_slice, affine_coords=temp_affine_coords, crop_coords=temp_crop_coords)
        x_shifted.append(temp_x_shifted)
        x_mask_shifted.append(temp_x_mask_shifted)

    x_shifted = np.array(x_shifted)
    x_mask_shifted = np.array(x_mask_shifted)

    return x_shifted, x_mask_shifted


def rigid_align_images(fixed_image, moving_image, fixed_mask=None, moving_mask=None, padding=10):
    fixed_cropped, fixed_mask_cropped = get_crop(fixed_image, x_mask=fixed_mask)
    moving_cropped, moving_mask_cropped = get_crop(moving_image, x_mask=moving_mask)

    target_size = min(fixed_cropped.shape, moving_cropped.shape)
    fixed_resize = skimage.transform.resize(fixed_cropped, output_shape=target_size, preserve_range=True, anti_aliasing=False)
    fixed_padded = np.pad(fixed_resize, pad_width=((padding, padding), (padding, padding)))

    fixed_mask_resize = skimage.transform.resize(fixed_mask_cropped, output_shape=target_size, preserve_range=True, anti_aliasing=False)
    fixed_mask_padded = np.pad(fixed_mask_resize, pad_width=((padding, padding), (padding, padding)))

    moving_resize = skimage.transform.resize(moving_cropped, output_shape=target_size, preserve_range=True, anti_aliasing=False)
    moving_padded = np.pad(moving_resize, pad_width=((padding, padding), (padding, padding)))

    moving_mask_resize = skimage.transform.resize(moving_mask_cropped, output_shape=target_size, preserve_range=True, anti_aliasing=False)
    moving_mask_padded = np.pad(moving_mask_resize, pad_width=((padding, padding), (padding, padding)))

    return fixed_padded, moving_padded, fixed_mask_padded, moving_mask_padded


def resize_and_crop(x, scale):
    # Makes the image bigger, but maintains the dimensions
    nx, ny = x.shape
    img_scaled = sktransform.rescale(x, scale)
    if scale > 1:
        new_center_x = int(nx * scale // 2)
        new_center_y = int(ny * scale // 2)
        resize_and_cropped = img_scaled[new_center_x - nx // 2:new_center_x + nx // 2,
                             new_center_y - ny // 2:new_center_y + ny // 2]
    elif scale < 1:
        pad_x = nx - int(nx * scale)
        pad_y = ny - int(ny * scale)
        pad_tuple = ((np.floor(pad_x / 2).astype(int), np.ceil(pad_x/2).astype(int)),
                     (np.floor(pad_y / 2).astype(int), np.ceil(pad_y/2).astype(int)))
        resize_and_cropped = np.pad(img_scaled, pad_width=pad_tuple)
    else:
        resize_and_cropped = x

    return resize_and_cropped


def rotate_around_p(x, degree, p):
    nx, ny = x.shape
    x_rot = np.zeros((nx, ny))
    # Theta should be in degrees
    rot_mat = rot2d(np.deg2rad(degree))
    x_coords = np.argwhere(x > 0)
    x_coords_rot = ((x_coords - p) @ rot_mat) + p
    ncoords = len(x_coords_rot)
    for i in range(ncoords):
        old_x, old_y = x_coords[i]
        rot_x, rot_y = x_coords_rot[i]
        x_value = x[old_x, old_y]
        rot_x = max(min(int(rot_x), nx - 1), 0)
        rot_y = max(min(int(rot_y), ny - 1), 0)
        x_rot[rot_x, rot_y] = x_value
    return x_rot

def optimize_rotation(x, y, p):
    # No interpolation is being done...
    # Here we optimize two masks...
    x = (x > 0).astype(int)
    y = (y > 0).astype(int)
    results = []
    for i_degree in range(360):
        x_rot = rotate_around_p(x, i_degree, p=p)
        x_rot = scipy.ndimage.binary_fill_holes(x_rot)
        score = (x_rot * y).sum()
        results.append((i_degree, score))

    sel_degree, max_score = sorted(results, key=lambda x: x[1])[-1]
    x_rot = rotate_around_p(x, sel_degree, p=p)
    x_rot = scipy.ndimage.binary_fill_holes(x_rot)
    return x_rot, sel_degree


def rot2d(theta):
    # Simple 2D rotation - assuming radians
    # theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s],  [s, c]])
    return rot_mat


def rot_x(degrees):
    # X-rotation matrix
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[1, 0, 0], [0, c, -s],  [0, s, c]])
    return rot_mat


def rot_y(degrees):
    # Y-rotation matrix
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return rot_mat


def rot_z(degrees):
    # Z-rotation matrix
    theta = np.radians(degrees)
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return rot_mat


def apply_shim(x, cpx_shim=None, n_chan=8):
    # cpx shim is the complex valued shim values
    if cpx_shim is None:
        amp = np.ones(n_chan)
        phase = np.random.normal(0, 0.5 * np.sqrt(np.pi), size=n_chan)
        cpx_shim = np.array([r * np.exp(1j * (phi + np.random.normal(0, 0.02 * np.sqrt(np.pi)))) for r, phi in zip(amp, phase)])

    x = np.einsum("tmn, t -> mn", x, cpx_shim)
    return x


def correct_mask_value(x, mask, atol=1.e-8):
    # Correct for -0 and 0 values that lie OUTSIDE the image.
    # Otherwise we get strange behaviour of the np.angle function
    # Find all the indices that are close to zero
    input_array_close = np.isclose(x, 0, atol=atol).astype(int)
    # But add on to that an additional condition to be outside of the mask
    mask_int = ((1 - mask) == 1).astype(int)
    # Create a mask that is both outside and close to zero
    x_outside = (input_array_close * mask_int).astype(bool)
    # Then set these to zero
    x[x_outside] = 0
    return x


def get_edge_mask(mask, outer_size, inner_size):
    outer_kernel = np.ones((outer_size, outer_size), np.uint8)
    outer_mask = cv2.erode(mask, outer_kernel, iterations=1)
    inner_kernel = np.ones((inner_size, inner_size), np.uint8)
    inner_mask = cv2.erode(mask, inner_kernel, iterations=1)
    edge_mask = outer_mask - inner_mask
    return edge_mask


def get_treshold_label_mask(x, structure=None, class_treshold=0.04, treshold_value=None, debug=False):
    # Class treshold: is a number in [0, 1], which value is used to treshold the size of each labeled region, which
    # are expressed as a parcentage. The labeled regions are the found continuous blobs of a certain size
    # The treshold value is optional, normally the mean is used to treshold the whole image
    # Method of choice
    if treshold_value is None:
        treshold_mask = x > (0.5*np.mean(x))
    else:
        treshold_mask = x > treshold_value

    treshold_mask = scipy.ndimage.binary_fill_holes(treshold_mask)

    if structure is None:
        structure = np.ones((3, 3))

    labeled, n_comp = scipy.ndimage.label(treshold_mask, structure)
    count_labels = [np.sum(labeled == i) for i in range(1, n_comp)]
    total_count = np.sum(count_labels)

    # If it is larger than 4%... then go (determined empirically)
    count_labels_index = [i + 1 for i, x in enumerate(count_labels) if x / total_count > class_treshold]
    if debug:
        print('Selected labels ', count_labels_index,'/', n_comp)
    if len(count_labels_index):
        x_mask = np.sum([labeled == x for x in count_labels_index], axis=0)
    else:
        x_mask = labeled == 1
    return x_mask


def get_otsu_mask(x, n_kernel=32):
    import SimpleITK as sitk
    import scipy.signal
    kernel_array = np.ones((n_kernel, n_kernel)) / n_kernel ** 2
    mask_smoothed = scipy.signal.convolve2d(x, kernel_array, 'same')

    inputImage = sitk.GetImageFromArray(mask_smoothed)
    inputImage = sitk.Cast(inputImage, sitk.sitkFloat32)

    maskImage = sitk.OtsuThreshold(inputImage, 0, 1, 50)
    mask_array = sitk.GetArrayFromImage(maskImage)
    mask_array = convex_hull_image(mask_array)
    return mask_array


def get_smoothed_mask(input_array, treshold_factor=0.1, treshold_smooth=0.9, n_mask=32, debug=False,
                      conv_mode='same', conv_boundary='symm'):
    """
    Get a smoothed mask based on the image values only

    TODO pretty strange that I first do the tresholding.. and THEN the smoothing.
    Might need to change that...

    :param input_array:
    :param treshold_factor: Factor of max absolute value of the array that is eventually as the cut off value
    :param treshold_smooth: Value used as treshold on the smoothed mask (by convolution)
    :param n_mask: Size of the kernel used in smoothing / convolving
    :param debug:
    :param conv_mode: option on convolve2d (from scipy.signal)
    :param conv_boundary: option on convolve2d (from scipy.signal)
    :return:
    """
    abs_summed = np.abs(input_array)
    treshhold = np.max(abs_summed) * treshold_factor
    c_tresh = (abs_summed > treshhold).astype(int)
    kernel = np.ones((n_mask, n_mask)) / n_mask ** 2
    smooth_mask = scipy.signal.convolve2d(c_tresh, kernel, mode=conv_mode, boundary=conv_boundary)
    c_tresh_smooth = (smooth_mask > treshold_smooth).astype(int)
    filled_mask = scipy.ndimage.binary_fill_holes(c_tresh_smooth)
    if debug:
        debug_dict = {'c_tresh': c_tresh,
                      'smooth_mask': smooth_mask,
                      'c_tresh_smooth': c_tresh_smooth}
        return debug_dict
    else:
        return filled_mask.astype(int)


def smooth_image(x, n_kernel=1, conv_mode='same', conv_boundary='symm'):
    kernel = np.ones((n_kernel, n_kernel)) / n_kernel ** 2
    x_smooth = scipy.signal.convolve2d(x, kernel, mode=conv_mode, boundary=conv_boundary)
    return x_smooth

smooth_gaussian_image = scipy.ndimage.gaussian_filter


def adaptive_smoothing_grid(input_image, smoothing_kernel_size):
    n_bins = len(smoothing_kernel_size)
    # smoothing_kernel_size SHould have the length of n_bins
    # n_bins is the number of bins the total range of magnitude_xx_yy is chopped up
    image_xx = np.diff(np.diff(input_image, axis=0), axis=0)[:, :-2]
    image_yy = np.diff(np.diff(input_image, axis=1), axis=1)[:-2, :]
    magnitude_xx_yy = np.sqrt(image_xx ** 2 + image_yy ** 2)
    magnitude_xx_yy = np.pad(magnitude_xx_yy, ((1, 1), (1, 1)))
    # Find the first quantile that is non zero.
    q0 = 0
    while np.percentile(magnitude_xx_yy, q0) == 0 and q0 < 100:
        q0 += 1
    bin_edges = [0] + [np.percentile(magnitude_xx_yy, x) for x in np.linspace(q0, 100, n_bins)]
    bin_factors = list(range(1, n_bins+1))
    temp = []
    for i in range(n_bins):
        lower = bin_edges[i] <= magnitude_xx_yy
        higher = magnitude_xx_yy < bin_edges[i + 1]
        temp.append(bin_factors[i] * lower * higher)
    input_image_smoothed_list = [smooth_image(input_image, x) for x in smoothing_kernel_size]
    input_image_smoothed = 0
    for i in range(n_bins):
        input_image_smoothed += (temp[i] > 0) * input_image_smoothed_list[i]
    return input_image_smoothed, temp


def lowpass_filter(x, p_kspace=0.49):
    # Removes the higher frequencies from an image
    n_y, n_x = x.shape
    limit_y = int(p_kspace * n_y)
    limit_x = int(p_kspace * n_x)
    print('Number of lines ', limit_y)
    x_kspace = transform_image_to_kspace_fftn(x)
    x_kspace[:limit_y] = 0
    x_kspace[-limit_y:] = 0
    x_kspace[:, :limit_x] = 0
    x_kspace[:, -limit_x:] = 0
    res = np.abs(transform_kspace_to_image_fftn(x_kspace))
    return res


def shrink_image(x, n_pixel, order=2):
    old_shape = x.shape
    new_shape = np.array(x.shape) - n_pixel
    x_resize = sktransf.resize(x, output_shape=new_shape, preserve_range=True, order=order)
    x_padded = np.pad(x_resize, [(n_pixel // 2, n_pixel // 2), (n_pixel // 2, n_pixel // 2)])
    # Got an error that talked about a size mismatch
    x_padded = sktransf.resize(x_padded, output_shape=old_shape, preserve_range=True)
    return x_padded

def convex_hull_image(data):
    region = np.argwhere(data)
    hull = ConvexHull(region)
    verts = [(region[v, 0], region[v, 1]) for v in hull.vertices]
    img = Image.new('L', data.shape, 0)
    ImageDraw.Draw(img).polygon(verts, outline=1, fill=1)
    mask = np.array(img)

    return mask.T, verts


def get_minmeanmediammax(x):
    # Very simple.. later added the median as well...
    return [np.min(x), np.mean(x), np.median(x), np.max(x)]


def get_minmax_cpx(x):
    x_abs = np.abs(x)
    x_ang = np.angle(x)
    x_real = np.real(x)
    x_imag = np.imag(x)
    names = ['min_abs', 'mean_abs', 'max_abs', 'min_angle', 'mean_angle', 'max_angle',
             'min_real', 'mean_real', 'max_real', 'min_imag', 'mean_imag', 'max_imag']
    values = [np.min(x_abs), np.mean(x_abs), np.max(x_abs), np.min(x_ang), np.mean(x_ang), np.max(x_ang),
              np.min(x_real), np.mean(x_real), np.max(x_real), np.min(x_imag), np.mean(x_imag), np.max(x_imag)]
    return dict(zip(names, values))


def print_min_max_cpx(x):
    dict_values = get_minmax_cpx(x)
    for k, v in dict_values.items():
        print(k, v)


def scale_mean_std(x, axis=None):
    x = (x - x.mean(axis=axis, keepdims=True)) / np.abs(x).std(axis=axis, keepdims=True)
    return x


def scale_median_std(x, axis=None):
    x = (x - np.median(x, axis=axis, keepdims=True)) / np.abs(x).std(axis=axis, keepdims=True)
    return x


def scale_minmax(x, is_complex=False, axis=None):
    # Added a check on when max and min are the same...
    # Otherwise we get a divide by zero.. and this way we might
    # be able to handle constant images.. (max==min)
    if is_complex:
        delta_max_min = (np.max(np.abs(x), axis=axis, keepdims=True) - np.min(np.abs(x), axis=axis, keepdims=True))
        return (x - np.min(np.abs(x), axis=axis, keepdims=True)) / delta_max_min
    else:
        delta_max_min = (np.max(x, axis=axis, keepdims=True) - np.min(x, axis=axis, keepdims=True))
        return (x - np.min(x, axis=axis, keepdims=True)) / delta_max_min


def scale_minmedian(x, is_complex=False):
    if is_complex:
        return (x - np.min(np.abs(x))) / (np.median(np.abs(x)) - np.min(np.abs(x)))
    else:
        return (x - np.min(x))/(np.median(x) - np.min(x))


def scale_minpercentile(x, q, is_complex=False, axis=None):
    # q ranges from 0..100
    if is_complex:
        return (x - np.min(np.abs(x), axis=axis, keepdims=True)) / (np.percentile(np.abs(x), q, axis=axis, keepdims=True) - np.min(np.abs(x), axis=axis, keepdims=True))
    else:
        return (x - np.min(x, axis=axis, keepdims=True))/(np.percentile(x, q, axis=axis, keepdims=True) - np.min(x, axis=axis, keepdims=True))


def treshold_percentile(x, q, is_complex=False, axis=None):
    # q ranges from 0..100
    if is_complex:
        perc_q = np.percentile(np.abs(x), q=q, axis=axis)
        # Hmmmmmm... not great
        x[np.abs(x) > perc_q] = perc_q
    else:
        perc_q = np.percentile(x, q=q)
        x[x > perc_q] = perc_q

    return x


def treshold_percentile_both(x, q, is_complex=False, axis=None):
    # q ranges from 0..100
    if is_complex:
        x = treshold_percentile(x, q)
        perc_q_bottom = np.percentile(np.abs(x), q=100-q, axis=axis)
        x[np.abs(x) < perc_q_bottom] = perc_q_bottom
    else:
        x = treshold_percentile(x, q)
        perc_q_bottom = np.percentile(x, q=100-q)
        x[x < perc_q_bottom] = perc_q_bottom

    return x


def scale_minpercentile_both(x, q, is_complex=False, axis=None):
    # q ranges from 0..100
    # Use the qth percentile as max instead of the max value...
    # ZO krijg je geen plaatje die van 0..1 geschaald is.. dat is zeker
    # Maar je kan hier na hm afkappen van 0..1
    if is_complex:
        x_min = np.percentile(np.abs(x), 100-q, axis=axis, keepdims=True)
        x_max = np.percentile(np.abs(x), q, axis=axis, keepdims=True)
        nominator = (x - x_min)
        denominator = (x_max - x_min)
        return nominator / denominator
    else:
        x_min = np.percentile(x, 100-q, axis=axis, keepdims=True)
        x_max = np.percentile(x, q, axis=axis, keepdims=True)
        nominator = (x - x_min)
        denominator = (x_max - x_min)
        return nominator / denominator


def get_clahe(x):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(2, 2))
    x = clahe.apply(x.astype(np.uint8))
    return x


def get_stiched(patches, target_shape, patch_shape, stride, _dtype=np.float):
    # Patch shape, assumed to have equal ndims as target shape
    # Currently only configured for 2d cases though..

    img_Y, img_X = target_shape
    patch_y, patch_x = patch_shape
    n_y, n_x = (int((img_Y - patch_y) / stride + 1), int((img_X - patch_x) / stride + 1))

    out_array = np.zeros(target_shape, dtype=_dtype)
    correction_array = np.zeros(target_shape, dtype=_dtype)
    counter = 0
    for ind_y in range(n_y):
        for ind_x in range(n_x):
            low_y = stride * ind_y
            high_y = low_y + patch_y
            low_x = stride * ind_x
            high_x = low_x + patch_x
            temp_img = out_array[low_y: high_y, low_x:high_x]

            # This is to be sure that we put in the right shape..
            if temp_img.shape == tuple([patch_y, patch_y]):
                out_array[low_y: high_y, low_x:high_x] += patches[counter]
                correction_array[low_y: high_y, low_x:high_x] += 1
                counter += 1

    return out_array / correction_array


def rotate_complex(x, angle=90, axes=(1, 0), reshape=True):
    # TODO THis is not needed anymore! (Scipy version > 1.6.0 )
    # Angle is always in degrees...
    real_trans_y = scipy.ndimage.rotate(np.real(x), angle=angle, axes=axes, reshape=reshape)
    imag_trans_y = scipy.ndimage.rotate(np.imag(x), angle=angle, axes=axes, reshape=reshape)
    # real_trans_y = scint.rotate(np.real(x), angle=angle, axes=axes, reshape=reshape)
    # imag_trans_y = scint.rotate(np.imag(x), angle=angle, axes=axes, reshape=reshape)
    temp_y = real_trans_y + 1j * imag_trans_y
    return temp_y


# Continue with this...
def get_patches(input_array, patch_shape, stride):
    # # source: https://gist.github.com/hasnainv/49dc4a85933de6b979f8

    temp_patches = skimage.util.view_as_windows(input_array, patch_shape, stride)
    n_y, n_x, patch_y, patch_x = temp_patches.shape
    # Get the prediction using this input
    input_patches = np.reshape(temp_patches, (n_y * n_x, patch_y, patch_x))
    return input_patches


def get_patches_cpx(x, patch_shape, stride, cpx_type='cartesian'):
    # We assume x is complex valued... hence, skimage cannot accept it.
    # We assume an input of (x,y)
    # and expect an output of (x, y) -> no stacked
    if cpx_type == 'cartesian':
        real_patch_x = get_patches(np.real(x), patch_shape, stride)
        imag_patch_x = get_patches(np.imag(x), patch_shape, stride)
        cpx_patch_x = real_patch_x + 1j * imag_patch_x
    elif cpx_type == 'polar':
        abs_patch_x = get_patches(np.abs(x), patch_shape, stride)
        ang_patch_x = get_patches(np.angle(x), patch_shape, stride)
        cpx_patch_x = abs_patch_x * np.exp(-1j * ang_patch_x)
    else:
        cpx_patch_x = -1
        warnings.warn('Unknown complex type, try `cartesian` or `polar`. Received {}'.format(cpx_type))
    return cpx_patch_x


def random_coil_comb(x):
    # Assume x of shape (n_chan, x, y)
    # And complex valued.
    n_phi, _, _ = x.shape  # n_phi is the amount of complex images we have.
    # I believe it does not really matter here.. since.. random... -..+...
    random_phi_input = np.array([np.exp(-1j * np.random.uniform(-np.pi, np.pi)) for _ in range(n_phi)])
    x_comb = (x.T.dot(random_phi_input)).T
    return x_comb


def to_complex_alternating(x, complex_type='cartesian'):
    # Assumes we have the complex channel in the first dimension, in an alternating fashion
    if x.shape[0] > 1:
        if complex_type == 'cartesian':
            x_cpx = x[::2] + 1j * x[1::2]
        elif complex_type == 'polar':
            x_cpx = x[::2] * np.exp(1j * x[1::2])
        else:
            print('unknown complex type:', complex_type)
            x_cpx = -1
    else:
        x_cpx = x

    return x_cpx


def to_complex(x, complex_type='cartesian', axis=-1):
    # Assume we have in the last two axis the complex number..
    if complex_type == 'cartesian':
        x_cpx = np.take(x, 0, axis=axis) + 1j * np.take(x, 1, axis=axis)
    elif complex_type == 'polar':
        x_cpx = np.take(x, 0, axis=axis) * np.exp(1j * np.take(x, 1, axis=axis))
    else:
        x_cpx = -1
        warnings.warn('Unknown complex type, try `cartesian` or `polar`. Received {}'.format(complex_type))
    return x_cpx


def to_complex_chan(x, img_shape=(512, 256), n_chan=8, complex_type='cartesian'):
    # When we have n_chan-coils, with complex values also stored in the last axis..
    # We can untangle this with this function
    # Input shape is something like (n_img, n_y, n_x, n_channels)
    # Output shape is a complex valued array of shape (n_img, n_y, n_x, n_channels//2)
    x_temp = np.squeeze(x).T
    x_temp = x_temp.reshape(2, n_chan, *img_shape[::-1], -1)
    x_cpx = to_complex(x_temp.T, complex_type=complex_type).T
    x_cpx = np.moveaxis(x_cpx.T, -1, 1)
    # x_cpx = np.squeeze(x_cpx.T)
    return x_cpx


def to_stacked(x, cpx_type='cartesian', stack_ax=-1):
    # We store the abs/angl or real/imag part in the last axis..
    if cpx_type == 'cartesian':
        x_stacked = np.stack([np.real(x), np.imag(x)], axis=stack_ax)
    elif cpx_type == 'polar':
        x_stacked = np.stack([np.abs(x), np.angle(x)], axis=stack_ax)
    else:
        x_stacked = -1
        warnings.warn('Unknown complex type, try `cartesian` or `polar`. Received {}'.format(cpx_type))
    return x_stacked


def random_transformation(x, **kwargs):
    # Assumes that we have input x (or also y)
    y = kwargs.get('y', np.empty(x.shape))
    debug = kwargs.get('debug')
    # Flip some stuff
    random_transf = np.random.randint(0, 3)
    if debug:
        print('Applying transformation ', random_transf)

    if random_transf == 0:
        x_shape = x.shape[0]
        x = np.take(x, range(x_shape)[::-1], axis=0)
        y = np.take(y, range(x_shape)[::-1], axis=0)
    elif random_transf == 1:
        x_shape = x.shape[1]
        x = np.take(x, range(x_shape)[::-1], axis=1)
        y = np.take(y, range(x_shape)[::-1], axis=1)
    elif random_transf == 2:
        degree_rot = np.random.randint(-10, 10)
        x = rotate_complex(x, angle=degree_rot)
        y = rotate_complex(y, angle=degree_rot)
    else:
        pass

    if np.isclose(np.sum(y), 0):
        return x
    else:
        return x, y


def point_to_line_dist(point, line):
    """Calculate the distance between a point and a line segment.

    To calculate the closest distance to a line segment, we first need to check
    if the point projects onto the line segment.  If it does, then we calculate
    the orthogonal distance from the point to the line.
    If the point does not project to the line segment, we calculate the
    distance to both endpoints and take the shortest distance.

    :param point: Numpy array of form [x,y], describing the point.
    :type point: numpy.core.multiarray.ndarray
    :param line: list of endpoint arrays of form [P1, P2]
    :type line: numpy array of line points
    :return: The minimum distance to a point.
    :rtype: float
    """
    # unit vector
    unit_line = line[1] - line[0]
    norm_unit_line = unit_line / np.linalg.norm(unit_line)

    # compute the perpendicular distance to the theoretical infinite line
    segment_dist = (
        np.linalg.norm(np.cross(line[1] - line[0], line[0] - point)) /
        np.linalg.norm(unit_line)
    )

    diff = (
        (norm_unit_line[0] * (point[0] - line[0][0])) +
        (norm_unit_line[1] * (point[1] - line[0][1]))
    )

    x_seg = (norm_unit_line[0] * diff) + line[0][0]
    y_seg = (norm_unit_line[1] * diff) + line[0][1]

    endpoint_dist = min(
        np.linalg.norm(line[0] - point),
        np.linalg.norm(line[1] - point)
    )

    # decide if the intersection point falls on the line segment
    lp1_x = line[0][0]  # line point 1 x
    lp1_y = line[0][1]  # line point 1 y
    lp2_x = line[1][0]  # line point 2 x
    lp2_y = line[1][1]  # line point 2 y
    is_betw_x = (lp1_x <= x_seg <= lp2_x) or (lp2_x <= x_seg <= lp1_x)
    is_betw_y = (lp1_y <= y_seg <= lp2_y) or (lp2_y <= y_seg <= lp1_y)
    if is_betw_x and is_betw_y:
        return segment_dist
    else:
        # if not, then return the minimum distance to the segment endpoints
        return endpoint_dist


def roll_zeropad(a, shift, axis=None):
    """
    Credits to:
    https://stackoverflow.com/questions/2777907/python-numpy-roll-with-padding
    Roll array elements along a given axis.

    Elements off the end of the array are treated as zeros.

    Parameters
    ----------
    a : array_like
        Input array.
    shift : int
        The number of places by which elements are shifted.
    axis : int, optional
        The axis along which elements are shifted.  By default, the array
        is flattened before shifting, after which the original
        shape is restored.

    Returns
    -------
    res : ndarray
        Output array, with the same shape as `a`.

    See Also
    --------
    roll     : Elements that roll off one end come back on the other.
    rollaxis : Roll the specified axis backwards, until it lies in a
               given position.
    """
    a = np.asanyarray(a)
    if shift == 0:
        return a
    if axis is None:
        n = a.size
        reshape = True
    else:
        n = a.shape[axis]
        reshape = False
    if np.abs(shift) > n:
        res = np.zeros_like(a)
    elif shift < 0:
        shift += n
        zeros = np.zeros_like(a.take(np.arange(n-shift), axis))
        res = np.concatenate((a.take(np.arange(n-shift,n), axis), zeros), axis)
    else:
        zeros = np.zeros_like(a.take(np.arange(n-shift,n), axis))
        res = np.concatenate((zeros, a.take(np.arange(n-shift), axis)), axis)
    if reshape:
        return res.reshape(a.shape)
    else:
        return res


def shuffle_array(input_array, model_dim, n_slice=1, randomness_perc=None, **kwargs):
    """
    Just randoms shuffles the rows of an array...

    The condition to separate the fourth dimension from the third... might not be necessary... but it is easier this
    way.
    Might also split this function... but that also seems... annoying..

    :param input_array: 2d or 3d array
    :param n_slice: determines the length of the slices
    :param randomness_perc:
    :return:
    """
    # Somehow, ignore the extra dimension at the end... in the normal case..
    # Or learn to deal with it.

    debug = kwargs.get('debug')

    # The axis over which we shuffle will always be the first...
    n_shuffle_axis = input_array.shape[0]

    assert n_shuffle_axis % n_slice == 0, "Error with dimension {n_dim}. \n Try these n_slice values {n_opt}".format(
        n_dim=input_array.shape, n_opt=', '.join(map(str, simple_div(n_shuffle_axis))))

    if debug and False:
        # Thus produces a lot of results.. not waiting for it
        print('INFO - SHF: \t Input array shape', input_array.shape)
        print('INFO - SHF: \t n shuffle axis', n_shuffle_axis)
        print('INFO - SHF: \t n slice', n_slice)

    # Group mapping
    # This determines the difficulty of the problem. More groups, is more complex
    group_index = np.arange(n_slice)
    # This is how often one group occurs.
    multiplicity_group = n_shuffle_axis // n_slice
    # Now we translate this into a mapping array. Where we see the relation between the groups and positions
    mapping_array = np.array(list(zip(np.repeat(group_index, multiplicity_group), np.arange(n_shuffle_axis))))  ## !!

    # Transform it to a dictionary
    mapping_dict = collections.defaultdict(list)
    [mapping_dict[key].append(val) for key, val in mapping_array]

    # Perform the actual shuffeling
    res_shuffle = np.copy(input_array)
    # Dimension cases..
    # 2D : (ny, nx)
    # 2D : (ny, nx, cpx)
    # --> shuffle over ny

    # 3D : (ntime, ny, nx)
    # 3D : (ntime, ny, nx, cpx)
    # --> for loop over ny and shuffle over ntime

    if model_dim == 2:
        # Shuffle the group indices
        shuffled_group = np.random.permutation(group_index)
        if randomness_perc:
            shuffled_group = de_randomize_loop(shuffled_group, randomness_perc)
        # Remapping
        shuffled_index = np.asarray(list(itertools.chain(*[mapping_dict[x] for x in shuffled_group])))
        res_shuffle = res_shuffle[shuffled_index, :]

    elif model_dim == 3:
        n_row = input_array.shape[1]
        # Because we are in three dimensions, we will shuffle each row of the data independently
        for i in range(n_row):
            # Shuffle the group indices
            shuffled_group = np.random.permutation(group_index)
            if randomness_perc:
                shuffled_group = de_randomize_loop(shuffled_group, randomness_perc)
            # Remapping
            shuffled_index = np.asarray(list(itertools.chain(*[mapping_dict[x] for x in shuffled_group])))
            res_shuffle[:, i] = res_shuffle[shuffled_index, i]
    else:
        print('No proper model_dim found, exit out of function. Found value: ', model_dim)

    return res_shuffle


def load_random_file(image_path, train=True):
    """
    Load some random extention.. with normalization

    :param image_path:
    :return:
    """

    most_common_ext = collections.Counter([os.path.splitext(x)[1] for x in os.listdir(image_path)]).most_common(1)
    ext = most_common_ext[0][0]

    list_files_ext = sorted([x for x in os.listdir(image_path) if x.endswith(ext)])

    if train:
        n_max = round(len(list_files_ext)*0.7)
        rfile_index = np.random.randint(n_max)
    else:
        n_max = round(len(list_files_ext) * 0.7, len(list_files_ext))
        rfile_index = np.random.randint(n_max)

    random_file = os.path.join(image_path, list_files_ext[rfile_index])
    if ext == '.dcm':
        input_image = pydicom.dcmread(random_file).pixel_array
    elif ext == '.npy':
        input_image = np.load(random_file)
    elif ext == '.png':
        input_image = np.array(Image.open(random_file))
    else:
        input_image = -1
        print('Extention not found', ext)

    input_image = input_image - np.mean(input_image)
    if np.std(input_image):
        # This can be zero...
        input_image /= np.std(input_image)

    return input_image


def de_randomize(x):
    x = np.array(list(x))
    dest_int = np.random.randint(len(x))
    found_int = np.where(x == dest_int)
    if len(found_int[0]):
        # print(found_int, len(found_int[0]))
        orig_int = found_int[0][0]
    else:
        orig_int = dest_int
    # Here we implicitly say where the integer needs to be.
    x[[orig_int, dest_int]] = x[[dest_int, orig_int]]

    return x, dest_int, orig_int


def de_randomize_loop(x_random, pc_goal, n_iter_max=200, eps=0.05):
    """
    This might need some explanation.. We try to measure and control HOW random something is.

    The measure for this is the distance from its original position.. and this is manipulated and normalized in such
    a way that we can probably use it over difference sequence length.
    It is normalized by the maximum amount of randomness we can achieve by this metric. This comes as a simple
    formula expressed as x_max_random.

    :param x_random:
    :param pc_goal:
    :param n_iter_max:
    :param eps:
    :return:
    """
    N = len(x_random)
    x_orig = np.arange(N)
    x_max_random = (N * (N - 1) / 2) / ((N - 1) * 2)
    x_temp, dest_int, orig_int = de_randomize(x_random)
    metric_random_init = (np.sum(np.abs(x_orig - np.argsort(x_temp))) / 2 - 1) / N + 1 / N
    metric_random = metric_random_init
    n_iter = 0
    c_goal = pc_goal * x_max_random

    while n_iter < n_iter_max and np.abs(metric_random - c_goal) >= eps:
        n_iter += 1
        if metric_random - c_goal < 0:
            # If we failed to get closer to the goal.. (However... is this properly done..?)
            # un-do this point, and try another
            # It can be shown that this method is far superior than keep trying random guesses.
            x_temp[[dest_int, orig_int]] = x_temp[[orig_int, dest_int]]
            x_temp, dest_int, orig_int = de_randomize(x_temp)
        else:
            # Switch two positions..
            x_temp, dest_int, orig_int = de_randomize(x_temp)

        # Check if the switching created a more or less random array...
        metric_random = (np.sum(np.abs(x_orig - np.argsort(x_temp))) / 2 - 1) / N + 1 / N

    return x_temp


def transform_kspace_to_image_fftn(k, dim=None, img_shape=None):
    """ Computes the Fourier transform from k-space to image space
    along a given or all dimensions
    :param k: k-space data
    :param dim: vector of dimensions to transform
    :param img_shape: desired shape of output image
    :returns: data in image space (along transformed dimensions)
    """
    if not dim:
        dim = range(k.ndim)

    img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(k, axes=dim), s=img_shape, axes=dim), axes=dim)
    img *= np.sqrt(np.prod(np.take(img.shape, dim)))
    return img


def transform_image_to_kspace_fftn(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)

    k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


def transform_image_to_kspace_fftn_torch(img, dim=None, k_shape=None):
    """ Computes the Fourier transform from image space to k-space space
    along a given or all dimensions
    :param img: image space data
    :param dim: vector of dimensions to transform
    :param k_shape: desired shape of output k-space data
    :returns: data in k-space (along transformed dimensions)
    """
    if not dim:
        dim = range(img.ndim)
    import torch
    import numpy as np
    import skimage.data
    import helper.plot_class as hplotc
    A = skimage.data.astronaut()[:, :, 0]
    # (batch, complex, nx, ny)
    A_tens = torch.from_numpy(np.stack([A, A], axis=-1))[None]/255.
    import torch.fft
    # Not sure if this works...
    # Okay eeehh. Only works for even sized matrices
    def roll_tensor(x):
        return torch.roll(x, shifts=[x // 2 for x in x.shape[-3:-1]], dims=(-3, -2))

    A_rolled = roll_tensor(A_tens)
    A_fftd = torch.fft.fft(A_rolled[0, :, :, 0])
    hplotc.ListPlot([A_fftd], augm='np.abs')
    A_returned = torch.fft.ifft(A_rolled, signal_ndim=2)
    hplotc.ListPlot([A_rolled, roll_tensor(A_rolled)])
    # k = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img, axes=dim), s=k_shape, axes=dim), axes=dim)
    # k /= np.sqrt(np.prod(np.take(img.shape, dim)))
    return k


def resize_complex_array(x, new_shape, preserve_range=True, anti_aliasing=False):
    x_real = sktransf.resize(x.real, new_shape, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    x_imag = sktransf.resize(x.imag, new_shape, preserve_range=preserve_range, anti_aliasing=anti_aliasing)

    return x_real + 1j * x_imag


def rescale_complex_array(x, scale, preserve_range=True, anti_aliasing=False):
    x_real = sktransf.rescale(x.real, scale=scale, preserve_range=preserve_range, anti_aliasing=anti_aliasing)
    x_imag = sktransf.rescale(x.imag, scale=scale, preserve_range=preserve_range, anti_aliasing=anti_aliasing)

    return x_real + 1j * x_imag


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def split_area(x, n_bins):
    # Splits the array x into n_bins based off the area under the function (sum)
    bin_total = np.sum(x)
    single_bin = bin_total / n_bins
    cut_off_points = []
    start_index = 0
    final_index = 1
    while len(cut_off_points) < n_bins + 1:
        # print('Currently at index', final_index, '/', len(x), len(cut_off_points))

        while (np.sum(x[start_index: final_index]) < single_bin) and (final_index < len(x)):
            final_index += 1

        current_error = np.abs(np.sum(x[start_index: final_index]) - single_bin)
        prev_error = np.abs(np.sum(x[start_index: final_index - 1]) - single_bin)
        if current_error > prev_error and final_index > 1:
            final_index -= 1

        # print('Current value ', final_index,  np.sum(x[start_index: final_index-1]))
        cut_off_points.append(final_index)
        start_index = final_index
        final_index = start_index + 1

    return np.array(cut_off_points)


def split_area_validate(x, cut_off, n_bins):
    bin_total = np.sum(x)
    single_bin = bin_total / n_bins
    for i in range(len(cut_off)-1):
        temp = np.sum(x[cut_off[i]:cut_off[i+1]])
        print(temp, single_bin, '\t', (temp - single_bin)/single_bin * 100)


def split_array_fourier_basis(input_img, n_bins, debug=False):
    output_abs_kspace = transform_image_to_kspace_fftn(input_img, dim=(-2, -1))
    n_y, n_x = output_abs_kspace.shape
    x_range = np.linspace(-n_x // 2, n_x // 2, n_x + 1)[:-1]
    y_range = np.linspace(-n_y // 2, n_y // 2, n_y + 1)[:-1]
    X, Y = np.meshgrid(x_range, y_range)
    full_ellipse = np.sqrt((X / (0.5 * n_x)) ** 2 + (Y / (0.5 * n_y)) ** 2)
    # Estimate offsets in a non-linear manner with the x-axis....
    x_index = np.arange(0, n_x // 2)
    y_index = np.arange(0, n_y // 2)
    y_offset = np.int(0.02 * n_y)
    x_target = np.abs(output_abs_kspace[n_y // 2 - y_offset:n_y // 2 + y_offset, :n_x // 2])
    x_target = np.mean(x_target, axis=0)
    x_offset = np.int(0.02 * n_x)
    y_target = np.abs(output_abs_kspace[:n_y // 2, n_x // 2 - x_offset:n_x // 2 + x_offset])
    y_target = np.mean(y_target, axis=1)

    optimization_fun = lambda t, a, b, c: a * np.exp(b * (t - c))
    try:
        coeff_obj = scipy.optimize.curve_fit(optimization_fun, x_index, x_target, p0=(0.1, 0.1, 0.1), maxfev=10000)
        coeff_x = coeff_obj[0]
        approx_fun = optimization_fun(x_index, *coeff_x)
    except RuntimeError:
        coeff_obj = scipy.optimize.curve_fit(optimization_fun, y_index, y_target, p0=(0.1, 0.1, 0.1), maxfev=10000)
        coeff_x = coeff_obj[0]
        approx_fun = optimization_fun(y_index, *coeff_x)

    cut_off_points = split_area(approx_fun[::-1], n_bins=n_bins)
    cut_off_points[-1] = 9000

    res_list = []
    prev_radius = None
    for i_radius in cut_off_points:
        i_radius = int(i_radius)
        if prev_radius is None:
            selection_circle = full_ellipse <= (i_radius / (n_x // 2))
            res = output_abs_kspace * selection_circle.astype(int)
        else:
            prev_selection_circle = full_ellipse <= (prev_radius / (n_x // 2))
            selection_circle = full_ellipse <= (i_radius / (n_x // 2))
            res = output_abs_kspace * (selection_circle.astype(int) - prev_selection_circle.astype(int))

        prev_radius = i_radius
        res_list.append(res)

    target_array = [transform_kspace_to_image_fftn(x, dim=(-2, -1)) for x in res_list]
    target_array = np.array(target_array)
    target_array = np.real(target_array)

    if debug:
        return target_array, res_list
    else:
        return target_array


def get_proper_scaled(x, patch_shape, stride=None):
    if stride is None:
        stride = patch_shape[0] // 2

    temp_patches = get_patches(x, patch_shape=patch_shape, stride=stride)
    vmax = np.max([x.mean() for x in temp_patches]) + np.mean([x.std() for x in temp_patches])

    return vmax


def get_proper_scaled_v2(x, patch_shape, stride=None):
    if isinstance(patch_shape, int):
        patch_shape = (patch_shape, patch_shape)

    temp_patches = get_patches(x, patch_shape=patch_shape, stride=stride)
    # Subset those with enough non-zero values in their patches
    temp_patches = np.array([x for x in temp_patches if (np.isclose(x, 0, atol=1e-5)).sum() / np.prod(x.shape) < 0.5])
    max_patches = temp_patches.max(axis=(-2, -1))
    max_mean_patches_index = [(i, x.mean()) for i, x in enumerate(max_patches) if x.mean() > max_patches.mean()]

    sorted_max_mean_patches_index = sorted(max_mean_patches_index, key=lambda x: x[1])
    n_max_sel = int(len(sorted_max_mean_patches_index) * 0.2)

    patch_index, _ = zip(*sorted_max_mean_patches_index[:n_max_sel])
    patch_index = list(patch_index)
    sel_patches = temp_patches[patch_index]

    max_value = sel_patches.mean() + 2 * sel_patches.std()
    return max_value


def simple_div(n):
    """
    Calculate the divisors of n. Used to split arrays into a groups of integer size, or to asses how
    large the batch size should be when training a model
    :param n:
    :return:
    """
    return [i for i in range(n, 0, -1) if n % i == 0]


def get_slices_h5_file(h5_file, key='data'):
    import h5py
    with h5py.File(h5_file, 'r') as f:
        temp = f[key]
        n_slice = temp.shape[0]
    return  n_slice


def flatten_dict(nested_dict):
    # Needed by nested dict
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict, column_name="0"):
    # From: https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    # df = df.unstack(level=-1)
    # df.columns = df.columns.map("{0[1]}".format)
    df.columns = [column_name]
    return df


def get_all_mid_slices(image_array, offset=(0, 0, 0)):
    image_shape = np.array(image_array.shape)//2
    image_ndim = image_array.ndim
    ax_combinations = list(itertools.combinations(range(image_ndim), 2))
    sliced_array = []
    for i_ax_comb in ax_combinations:
        slice_list = [slice(None)] * image_ndim
        remaining_ax = set(range(image_ndim)).difference(set(i_ax_comb))
        # Create a list of slices..
        for i_ax in list(remaining_ax):
            slice_list[i_ax] = image_shape[i_ax] + offset[i_ax]
        sliced_array.append(image_array[tuple(slice_list)])
    return sliced_array


if __name__ == "__main__":
    import numpy as np
    import skimage.data
    import helper.plot_class as hplotc

    A = skimage.data.astronaut()[:, :, 0]
    A_kspace = transform_image_to_kspace_fftn(A)
    B = shuffle_array(A, 2, n_slice=32)
    B_kspace = shuffle_array(A_kspace, 2, n_slice=32)
    A_img_space = transform_kspace_to_image_fftn(B_kspace)
    hplotc.ListPlot([A, B, np.abs(B_kspace), np.abs(A_img_space)])
