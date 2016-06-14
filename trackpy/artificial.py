from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import numpy as np
from scipy.spatial import cKDTree
from trackpy.utils import validate_tuple


def draw_point(image, pos, value):
    image[tuple(pos)] = value


def feat_gauss(r):
    """ Gaussian at r = 0. """
    return np.exp(r**2 * r.ndim/-2)


def feat_ring(r, thickness):
    """ Ring feature with a gaussian profile with a certain thickness."""
    return np.exp(((r-1)/thickness)**2 * r.ndim/-2)


def feat_hat(r, disc_size):
    """ Solid disc of size disc_size, with Gaussian smoothed borders. """
    result = np.ones_like(r)
    mask = r > disc_size
    result[mask] = np.exp(((r[mask] - disc_size)/(1 - disc_size))**2 *
                          result.ndim/-2)
    return result


def feat_step(r):
    """ Solid disc. """
    return (r <= 1).astype(np.float)


def draw_feature(image, position, size, max_value=None, feat_func=feat_gauss,
                 ecc=None, mask_diameter=None, **kwargs):
    """ Draws a radial symmetric feature and adds it to the image at given
    position. The given function will be evaluated at each pixel coordinate,
    no averaging or convolution is done.

    Parameters
    ----------
    image : ndarray
        image to draw features on
    position : iterable
        coordinates of feature position
    size : number
        the size of the feature (meaning depends on feature, for feat_gauss,
        it is the radius of gyration)
    max_value : number
        maximum feature value. should be much less than the max value of the
        image dtype, to avoid pixel wrapping at overlapping features
    feat_func : function. Default: feat_gauss
        function f(r) that takes an ndarray of radius values
        and returns intensity values <= 1
    ecc : positive number, optional
        eccentricity of feature, defined only in 2D. Identical to setting
        diameter to (diameter / (1 - ecc), diameter * (1 - ecc))
    mask_diameter :
        defines the box that will be drawn on. Default 8 * size.
    kwargs : keyword arguments are passed to feat_func
    """
    if len(position) != image.ndim:
        raise ValueError("Number of position coordinates should match image"
                         " dimensionality.")
    size = validate_tuple(size, image.ndim)
    if ecc is not None:
        if len(size) != 2:
            raise ValueError("Eccentricity is only defined in 2 dimensions")
        if size[0] != size[1]:
            raise ValueError("Diameter is already anisotropic; eccentricity is"
                             " not defined.")
        size = (size[0] / (1 - ecc), size[1] * (1 - ecc))
    if mask_diameter is None:
        mask_diameter = tuple([s * 8 for s in size])
    else:
        mask_diameter = validate_tuple(mask_diameter, image.ndim)
    if max_value is None:
        max_value = np.iinfo(image.dtype).max - 3
    rect = []
    vectors = []
    for (c, s, m, lim) in zip(position, size, mask_diameter, image.shape):
        if (c >= lim) or (c < 0):
            raise ValueError("Position outside of image.")
        lower_bound = max(int(np.floor(c - m / 2)), 0)
        upper_bound = min(int(np.ceil(c + m / 2 + 1)), lim)
        rect.append(slice(lower_bound, upper_bound))
        vectors.append(np.arange(lower_bound - c, upper_bound - c) / s)
    coords = np.meshgrid(*vectors, indexing='ij', sparse=True)
    r = np.sqrt(np.sum(np.array(coords)**2, axis=0))
    spot = max_value * feat_func(r, **kwargs)
    image[rect] += spot.astype(image.dtype)


def gen_random_locations(shape, count, margin=0):
    """ Generates `count` number of positions within `shape`. If a `margin` is
    given, positions will be inside this margin. Margin may be tuple-valued.
    """
    margin = validate_tuple(margin, len(shape))
    np.random.seed(0)
    pos = [np.random.uniform(round(m), round(s - m), count)
           for (s, m) in zip(shape, margin)]
    return np.array(pos).T


def eliminate_overlapping_locations(f, separation):
    """ Makes sure that no position is within `separation` from each other, by
    deleting one of the that are to close to each other.
    """
    separation = validate_tuple(separation, f.shape[1])
    assert np.greater(separation, 0).all()
    # Rescale positions, so that pairs are identified below a distance of 1.
    f = f / separation
    while True:
        duplicates = cKDTree(f, 30).query_pairs(1)
        if len(duplicates) == 0:
            break
        to_drop = []
        for pair in duplicates:
            to_drop.append(pair[1])
        f = np.delete(f, to_drop, 0)
    return f * separation


def gen_nonoverlapping_locations(shape, count, separation, margin=0):
    """ Generates `count` number of positions within `shape`, that have minimum
    distance `separation` from each other. The number of positions returned may
    be lower than `count`, because positions too close to each other will be
    deleted. If a `margin` is given, positions will be inside this margin.
    Margin may be tuple-valued.
    When `subpx`=False, only returns integer coordinates.
    """
    positions = gen_random_locations(shape, count, margin)
    return eliminate_overlapping_locations(positions, separation)


def draw_spots(shape, positions, size, noise_level=0, bitdepth=8,
               feat_func=feat_gauss, ecc=None, mask_diameter=None, **kwargs):
    """ Generates an image with features at given positions. A feature with
    position x will be centered around pixel x. In other words, the origin of
    the output image is located at the center of pixel (0, 0).

    Parameters
    ----------
    shape : tuple of int
        the shape of the produced image
    positions : iterable of tuples
        an iterable of positions
    size : number
        the size of the feature (meaning depends on feature, for feat_gauss,
        it is the radius of gyration)
    noise_level : int, default: 0
        white noise will be generated up to this level
    bitdepth : int, default: 8
        the desired bitdepth of the image (<=32 bits)
    feat_func : function, default: feat_gauss
        function f(r) that takes an ndarray of radius values
        and returns intensity values <= 1
    ecc : positive number, optional
        eccentricity of feature, defined only in 2D. Identical to setting
        diameter to (diameter / (1 - ecc), diameter * (1 - ecc))
    mask_diameter :
        defines the box that will be drawn on. Default 8 * size.
    kwargs : keyword arguments are passed to feat_func
    """
    if bitdepth <= 8:
        dtype = np.uint8
        internaldtype = np.uint16
    elif bitdepth <= 16:
        dtype = np.uint16
        internaldtype = np.uint32
    elif bitdepth <= 32:
        dtype = np.uint32
        internaldtype = np.uint64
    else:
        raise ValueError('Bitdepth should be <= 32')
    np.random.seed(0)
    image = np.random.randint(0, noise_level + 1, shape).astype(internaldtype)
    for pos in positions:
        draw_feature(image, pos, size, max_value=2**bitdepth - 1,
                     feat_func=feat_func, ecc=ecc, mask_diameter=mask_diameter,
                     **kwargs)
    return image.clip(0, 2**bitdepth - 1).astype(dtype)
