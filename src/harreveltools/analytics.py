from typing import List, Tuple
import numpy as np
import scipy.ndimage


def lower_bound_line(points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[int]]:
    # Sort the points by x coordinate
    points.sort(key=lambda p: p[0])
    # Initialize a list to store the points on the lower bound line
    lower_bound = []
    index_lower_bound = []
    # Add the first point to the lower bound line
    lower_bound.append(points[0])
    # Iterate through the remaining points
    for i in range(1, len(points)):
        # Get the current point and the previous point on the lower bound line
        p, prev = points[i], lower_bound[-1]
        # If the y coordinate of the current point is less than the y coordinate of the previous point on the lower bound line,
        # add the current point to the lower bound line
        if p[1] < prev[1]:
            lower_bound.append(p)
            # Also store the index, so that we can back-track where the results came from
            index_lower_bound.append(i)
    # Return the lower bound line
    return lower_bound, index_lower_bound


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


def get_maximum_curvature(x_coord, y_coord):
    line_coords = np.stack([x_coord, y_coord])
    curvature = get_curvature(line_coords.T)
    index_max = np.argmax(curvature)
    return curvature, index_max


def get_curvature(x):
    # From a set of 2-D points in X
    # https: // en.wikipedia.org / wiki / Curvature
    dx_dt = np.gradient(x[:, 0])
    dy_dt = np.gradient(x[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt ** 2 + dy_dt ** 2) ** 1.5
    return curvature