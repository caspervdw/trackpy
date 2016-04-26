from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import warnings
import logging

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from .utils import validate_tuple

from .masks import binary_mask

logger = logging.getLogger(__name__)


def where_close(pos, separation, intensity=None):
    """ Returns indices of features that are closer than separation from other
    features. When intensity is given, the one with the lowest intensity is
    returned: else the most topleft is returned (to avoid randomness)"""
    if len(pos) == 0:
        return []
    separation = validate_tuple(separation, pos.shape[1])
    if any([s == 0 for s in separation]):
        return []
    # Rescale positions, so that pairs are identified below a distance
    # of 1.
    pos_rescaled = pos / separation
    duplicates = cKDTree(pos_rescaled, 30).query_pairs(1)
    if len(duplicates) == 0:
        return []
    index_0 = np.fromiter((x[0] for x in duplicates), dtype=int)
    index_1 = np.fromiter((x[1] for x in duplicates), dtype=int)
    if intensity is None:
        to_drop = np.where(np.sum(pos_rescaled[index_0], 1) >
                           np.sum(pos_rescaled[index_1], 1),
                           index_1, index_0)
    else:
        intensity_0 = intensity[index_0]
        intensity_1 = intensity[index_1]
        to_drop = np.where(intensity_0 > intensity_1, index_1, index_0)
        edge_cases = intensity_0 == intensity_1
        if np.any(edge_cases):
            index_0 = index_0[edge_cases]
            index_1 = index_1[edge_cases]
            to_drop[edge_cases] = np.where(np.sum(pos_rescaled[index_0], 1) <
                                           np.sum(pos_rescaled[index_1], 1),
                                           index_1, index_0)
    return np.unique(to_drop)


def drop_close(pos, separation, intensity=None):
    """ Removes features that are closer than separation from other features.
    When intensity is given, the one with the lowest intensity is dropped:
    else the most topleft is dropped (to avoid randomness)"""
    to_drop = where_close(pos, separation, intensity)
    return np.delete(pos, to_drop, axis=0)


def percentile_threshold(image, percentile):
    """Find grayscale threshold based on distribution in image."""

    not_black = image[np.nonzero(image)]
    if len(not_black) == 0:
        return np.nan
    return np.percentile(not_black, percentile)


def grey_dilation(image, separation, percentile=64, margin=None):
    """Find local maxima whose brightness is above a given percentile.

    Parameters
    ----------
    separation : minimum separation between maxima
    percentile : chooses minimum grayscale value for a local maximum
    margin : zone of exclusion at edges of image. Defaults to separation.
            A smarter value is set by locate().
    """
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError("Perform dilation on exact (i.e., integer) data.")

    ndim = image.ndim
    separation = validate_tuple(separation, ndim)
    if margin is None:
        margin = tuple([int(s / 2) for s in separation])

    # Compute a threshold based on percentile.
    threshold = percentile_threshold(image, percentile)
    if np.isnan(threshold):
        warnings.warn("Image is completely black.", UserWarning)
        return np.empty((0, ndim))

    # Find the smallest box that fits inside the ellipse given by separation
    size = [int(s / np.sqrt(ndim)) for s in separation]

    # The intersection of the image with its dilation gives local maxima.
    dilation = ndimage.grey_dilation(image, size, mode='constant')
    maxima = (image == dilation) & (image > threshold)
    if np.sum(maxima) == 0:
        warnings.warn("Image contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    pos = np.vstack(np.where(maxima)).T

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((pos < margin) | (pos > (shape - margin - 1)), 1)
    pos = pos[~near_edge]

    if len(pos) == 0:
        warnings.warn("All local maxima were in the margins.", UserWarning)
        return np.empty((0, ndim))

    # Remove local maxima that are too close to each other
    pos = drop_close(pos, separation, image[maxima][~near_edge])

    return pos


def grey_dilation_legacy(image, radius, percentile=64, margin=None):
    """Find local maxima whose brightness is above a given percentile.

    Parameters
    ----------
    radius : integer definition of "local" in "local maxima"
    percentile : chooses minimum grayscale value for a local maximum
    margin : zone of exclusion at edges of image. Defaults to radius.
            A smarter value is set by locate().
    """
    if margin is None:
        margin = radius

    ndim = image.ndim
    # Compute a threshold based on percentile.
    threshold = percentile_threshold(image, percentile)
    if np.isnan(threshold):
        warnings.warn("Image is completely black.", UserWarning)
        return np.empty((0, ndim))

    # The intersection of the image with its dilation gives local maxima.
    if not np.issubdtype(image.dtype, np.integer):
        raise TypeError("Perform dilation on exact (i.e., integer) data.")
    footprint = binary_mask(radius, ndim)
    dilation = ndimage.grey_dilation(image, footprint=footprint,
                                     mode='constant')
    maxima = np.vstack(np.where((image == dilation) & (image > threshold))).T
    if not np.size(maxima) > 0:
        warnings.warn("Image contains no local maxima.", UserWarning)
        return np.empty((0, ndim))

    # Do not accept peaks near the edges.
    shape = np.array(image.shape)
    near_edge = np.any((maxima < margin) | (maxima > (shape - margin - 1)), 1)
    maxima = maxima[~near_edge]
    if not np.size(maxima) > 0:
        warnings.warn("All local maxima were in the margins.", UserWarning)

    # Return coords in as a numpy array shaped so it can be passed directly
    # to the DataFrame constructor.
    return maxima
