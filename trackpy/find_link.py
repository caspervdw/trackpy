from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
from six.moves import range, zip
import warnings
import logging
import itertools, functools

import numpy as np
import pandas as pd
from scipy import ndimage

from .masks import slice_image, mask_image
from .find import grey_dilation, drop_close
from .utils import (default_pos_columns, guess_pos_columns, is_isotropic,
                    cKDTree, validate_tuple, pandas_sort)
from .try_numba import NUMBA_AVAILABLE
from .preprocessing import bandpass
from .refine import refine_com
from .linking import (Point, TrackUnstored, SubnetLinker, _points_to_arr,
                      SubnetOversizeException, nonrecursive_link, numba_link)
from .feature import characterize

logger = logging.getLogger(__name__)


def coords_from_df(df, pos_columns, t_column):
    """A generator that returns ndarrays of coords from a DataFrame. Assumes
    t_column to be of integer type. Float-typed integers are also accepted.

    Empty frames will be returned as empty arrays of shape (0, ndim)."""
    ndim = len(pos_columns)
    grouped = iter(df.groupby(t_column))  # groupby sorts by default

    # get the first frame to learn first frame number
    cur_frame, frame = next(grouped)
    cur_frame = int(cur_frame)
    yield cur_frame, frame[pos_columns].values
    cur_frame += 1

    for frame_no, frame in grouped:
        frame_no = int(frame_no)
        while cur_frame < frame_no:
            yield cur_frame, np.empty((0, ndim))
            cur_frame += 1

        yield cur_frame, frame[pos_columns].values
        cur_frame += 1


def coords_from_df_iter(df_iter, pos_columns, t_column):
    """A generator that returns ndarrays of coords from a generator of
    DataFrames. Also returns the first value of the t_column."""
    ndim = len(pos_columns)

    for df in df_iter:
        if len(df) == 0:
            yield None, np.empty((0, ndim))
        else:
            yield df[t_column].iloc[0], df[pos_columns].values


def link_simple_iter(coords_iter, search_range, **kwargs):
    """Link an iterable of per-frame coordinates into trajectories.

    Parameters
    ----------
    coords_iter : iterable or enumerated iterable of 2d numpy arrays
    search_range : float or tuple
    memory : integer
    predictor : predictor function; see 'predict' module

    Returns
    -------
    yields tuples (t, list of particle ids)
    """
    # ensure that coords_iter is iterable
    coords_iter = iter(coords_iter)

    # interpret the first element of the iterable
    val = next(coords_iter)
    if isinstance(val, np.ndarray):
        # the iterable was not enumerated, so enumerate the remainder
        coords_iter = enumerate(coords_iter, start=1)
        t, coords = 0, val
    else:
        t, coords = val

    #  obtain dimensionality
    ndim = coords.shape[1]
    search_range = validate_tuple(search_range, ndim)

    # initialize the linker and yield the particle ids of the first frame
    linker = Linker(search_range, **kwargs)
    linker.init_level(coords, t)
    yield t, linker.particle_ids

    for t, coords in coords_iter:
        linker.next_level(coords, t)
        yield t, linker.particle_ids


def link_simple(f, search_range, pos_columns=None, t_column='frame', **kwargs):
    """Link a DataFrame of coordinates into trajectories.

    Parameters
    ----------
    f : DataFrame containing feature positions and frame indices
    search_range : float or tuple
    memory : integer, optional
    pos_columns : list of str, optional
    t_column : str, optional

    Returns
    -------
    DataFrame with added column 'particle' containing trajectory labels.
    The t_column (by default: 'frame') will be coerced to integer."""
    if pos_columns is None:
        pos_columns = guess_pos_columns(f)
    ndim = len(pos_columns)
    search_range = validate_tuple(search_range, ndim)

    # copy the dataframe
    f = f.copy()
    # coerce t_column to integer type
    if not np.issubdtype(f[t_column].dtype, np.integer):
        f[t_column] = f[t_column].astype(np.integer)
    # sort on the t_column
    pandas_sort(f, t_column, inplace=True)

    coords_iter = coords_from_df(f, pos_columns, t_column)
    ids = []
    for i, _ids in link_simple_iter(coords_iter, search_range, **kwargs):
        ids.extend(_ids)

    f['particle'] = ids
    return f


def link_simple_df_iter(f_iter, search_range, pos_columns=None,
                        t_column='frame', **kwargs):
    """Link an iterable of DataFrames into trajectories.

    Parameters
    ----------
    f_iter : iterable of DataFrames with feature positions, frame indices
    search_range : float or tuple
    memory : integer, optional
    pos_columns : list of str, optional
    t_column : str, optional
    predictor : predictor function; see 'predict' module

    Yields
    -------
    DataFrames with added column 'particle' containing trajectory labels
    """
    if pos_columns is None:
        # Get info about the first frame without processing it
        f_iter, f_iter_dummy = itertools.tee(f_iter)
        f0 = next(f_iter_dummy)
        pos_columns = guess_pos_columns(f0)
        del f_iter_dummy, f0
    ndim = len(pos_columns)
    search_range = validate_tuple(search_range, ndim)

    f_iter, f_coords_iter = itertools.tee(f_iter)
    coords_iter = coords_from_df_iter(f_coords_iter, pos_columns, t_column)

    ids_iter = (_ids for _i, _ids in
        link_simple_iter(coords_iter, search_range, **kwargs))
    for df, ids in zip(f_iter, ids_iter):
        df_linked = df.copy()
        df_linked['particle'] = ids
        yield df_linked


def find_link(reader, search_range, separation, diameter=None, memory=0,
              minmass=0, noise_size=1, smoothing_size=None, threshold=None,
              percentile=64, before_link=None, after_link=None, refine=False,
              **kwargs):
    """Find and link features in an image sequence

    Parameters
    ----------
    reader : pims.FramesSequence
    search_range : number or tuple
        maximum displacement of features between subsequent frames
    separation : number or tuple
        minimum separation distance between features
    diameter : number or tuple, optional
        feature diameter, used for characterization only.
        Also determines the margin (margin = diameter // 2).
        Default: ``separation``.
    memory : number, optional
        number of frames that features are allowed to disappear. Experimental.
        Default 0.
    minmass : number, optional
        minimum integrated intensity (in masked image). Default 0.
    noise_size : number or tuple, optional
        Size of Gaussian kernel with whith the image is convoluted for noise
        reduction. Default 1.
    smoothing_size : number or tuple, optional
        Size of rolling average box for background subtraction.
        By default, equals ``separation``. This may introduce bias when refined on
        the background subtracted image!
    threshold : number, optional
        Threshold value for image. Default None.
    percentile : number, optional
        The upper percentile of intensities in the image are considered as
        feature locations. Default 64.
    before_link : function, optional
        This function is executed after the initial find of each frame, but
        but before the linking and relocating.
        It should take the following arguments (or ``**kwargs``):

        - ``coords``: `ndarray``containing the initially found feature coordinates
        - ``reader``: unprocessed reader (for access to other frames)
        - ``image``: unprocessed image
        - ``image_proc``: the processed image
        - ``diameter``
        - ``separation``
        - ``search_range``
        - ``margin``
        - ``minmass``

        It should return an ndarray of the same shape as ``coords``.
    after_link : function, optional
        This function is executed after the find and link of each frame. It
        should not change the number of features.
        It should take the following arguments (or ``**kwargs``):

        - ``features``: a DataFrame containing the feature coordinates and characterization.
        - ``reader``: unprocessed reader (for access to other frames)
        - ``image``: unprocessed image
        - ``image_proc``: the processed image
        - ``diameter``
        - ``separation``
        - ``search_range``
        - ``margin``
        - ``minmass``

        It should return a DataFrame like ``features``.
    refine : boolean, optional
        Convenience parameter to do center-of-mass refinement. Cannot be used
        combined with an ``after_link`` function. Default False.

    Notes
    -----
    This feature is a recent addition to trackpy that is still in its
    experimental phase. Please report any issues you encounter on Github.

    If you use this specific algorithm for your scientific publications, please
    mention the accompanying publication [1]_

    References
    ----------
    .. [1] van der Wel C., Kraft D.J. Automated tracking of colloidal clusters
    with sub-pixel accuracy and precision. J. Phys. Condens. Mat. 29:44001 (2017)
    DOI: http://dx.doi.org/10.1088/1361-648X/29/4/044001
    """
    shape = reader[0].shape
    ndim = len(shape)
    if smoothing_size is None:
        smoothing_size = separation
    smoothing_size = validate_tuple(smoothing_size, ndim)
    smoothing_size = tuple([int(s) for s in smoothing_size])
    separation = validate_tuple(separation, ndim)
    if diameter is None:
        diameter = separation
    else:
        diameter = validate_tuple(diameter, ndim)

    if refine:
        if after_link is not None:
            raise ValueError("Refine cannot be used together with after_link.")
        pos_columns = default_pos_columns(ndim)
        refine_columns = pos_columns[::-1] + ['mass']
        radius = tuple([d // 2 for d in diameter])
        def after_link(image, features, image_proc, **kwargs):
            coords = features[pos_columns].values
            if len(coords) == 0:
                return features
            # no separation filtering, because we use precise grey dilation
            coords = refine_com(image, image_proc, radius, coords, separation=0,
                                characterize=False)
            features[refine_columns] = coords
            return features

    features = []
    proc_func = lambda x: bandpass(x, noise_size, smoothing_size, threshold)
    generator = find_link_iter(reader, search_range, separation,
                               diameter=diameter, memory=memory,
                               percentile=percentile, minmass=minmass,
                               proc_func=proc_func, before_link=before_link,
                               after_link=after_link, **kwargs)
    for frame_no, f_frame in generator:
        if f_frame is None:
            n_traj = 0
        else:
            n_traj = len(f_frame)
        logger.info("Frame {0}: {1} trajectories present.".format(frame_no,
                                                                  n_traj))
        if n_traj == 0:
            continue
        features.append(f_frame)

    features = pd.concat(features, ignore_index=False)
    return features


def find_link_iter(reader, search_range, separation, diameter=None,
                   percentile=64, minmass=0, proc_func=None, before_link=None,
                   after_link=None, **kwargs):

    shape = reader[0].shape
    ndim = len(shape)

    search_range = validate_tuple(search_range, ndim)
    separation = validate_tuple(separation, ndim)
    isotropic = is_isotropic(diameter)
    if proc_func is None:
        proc_func = lambda x: x

    if diameter is None:
        diameter = separation
    else:
        diameter = validate_tuple(diameter, ndim)
    radius = tuple([int(d // 2) for d in diameter])
    # Define zone of exclusion at edges of image, avoiding features with
    # incomplete image data ("radius")
    margin = radius

    # Check whether the margins are not covering the complete image
    if np.any([s <= 2*m for (s, m) in zip(shape, margin)]):
        # Check whether the image looks suspiciously like a multichannel image.
        if np.any([s <= 4 for s in shape]) and (ndim > 2):
            raise ValueError('One of the image dimensions is very small. '
                             'Please make sure that you are not using an RGB '
                             'or other multichannel (color) image.')
        else:
            raise ValueError('The feature finding margins are larger than the '
                             'image shape. Please use smaller radius, '
                             'separation or smoothing_size.')

    linker = FindLinker(search_range, separation, diameter, minmass,
                        percentile, **kwargs)

    reader_iter = iter(reader)
    image = next(reader_iter)
    image_proc = proc_func(image)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coords = grey_dilation(image_proc, separation, percentile, margin,
                               precise=True)
    if before_link is not None:
        coords = before_link(coords=coords, reader=reader, image=image,
                             image_proc=image_proc,
                             diameter=diameter, separation=separation,
                             search_range=search_range,
                             margin=margin, minmass=minmass)
    extra_data = characterize(coords, image, radius)
    mask = extra_data['mass'] >= minmass
    coords = coords[mask]
    for key in extra_data:
        extra_data[key] = extra_data[key][mask]
    linker.init_level(coords, image.frame_no, extra_data)
    features = linker.coords_df
    if after_link is not None and features is not None:
        features = after_link(features=features, reader=reader, image=image,
                              image_proc=image_proc,
                              diameter=diameter, separation=separation,
                              search_range=search_range, margin=margin,
                              minmass=minmass)
        linker.coords_df = features  # for next iteration

    yield image.frame_no, features

    for image in reader_iter:
        image_proc = proc_func(image)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = grey_dilation(image_proc, separation, percentile, margin,
                                   precise=True)
        if before_link is not None:
            coords = before_link(coords=coords, reader=reader, image=image,
                                 image_proc=image_proc,
                                 diameter=diameter, separation=separation,
                                 search_range=search_range,
                                 margin=margin, minmass=minmass)
        extra_data = characterize(coords, image, radius)
        mask = extra_data['mass'] >= minmass
        coords = coords[mask]
        for key in extra_data:
            extra_data[key] = extra_data[key][mask]
        linker.next_level(coords, image.frame_no, image=image_proc,
                          extra_data=extra_data)
        features = linker.coords_df
        if after_link is not None and features is not None:
            features = after_link(features=features, reader=reader, image=image,
                                  image_proc=image_proc,
                                  diameter=diameter, separation=separation,
                                  search_range=search_range, margin=margin,
                                  minmass=minmass)
            linker.coords_df = features  # for next iteration
        yield image.frame_no, features


class PointFindLink(Point):
    """  Version of :class:`trackpy.linking.PointND` that is used for find_link.
    """
    def __init__(self, t, pos, id=None, extra_data=None):
        super(PointFindLink, self).__init__()
        self.t = t
        self.pos = np.asarray(pos)
        self.id = id
        if extra_data is None:
            self.extra_data = dict()
        else:
            self.extra_data = extra_data
        # self.back_cands = []
        self.forward_cands = []
        self.subnet = None
        self.relocate_neighbors = []


def _default_coord_mapping(search_range, level):
    """ Convert a list of Points to an ndarray of coordinates """
    return _points_to_arr(level) / search_range


def _wrap_predictor(search_range, predictor, t):
    """ Create a function that maps coordinates using a predictor class."""
    def coord_mapping(level):
        # swap axes order (need to do inplace to preserve the Point attributes)
        for p in level:
            p.pos = p.pos[::-1]
        result = np.array(list(predictor(t, level)))
        for p in level:  # swap axes order back
            p.pos = p.pos[::-1]
        return result[:, ::-1] / search_range

    return coord_mapping


class TreeFinder(object):
    def __init__(self, points, search_range):
        """Takes a list of particles."""
        self.ndim = len(search_range)
        self.search_range = np.atleast_2d(search_range)
        if not isinstance(points, list):
            points = list(points)
        self.points = points
        self.set_predictor(None)
        self.rebuild()

    def __len__(self):
        return len(self.points)

    def add_point(self, pt):
        self.points.append(pt)
        self._clean = False

    def set_predictor(self, predictor, t=None):
        """Sets a predictor to the TreeFinder

        predictor : function, optional

            Called with t and a list of N Point instances, returns their
            "effective" locations, as an N x d array (or any iterable).
            Used for prediction (see "predict" module).
        """
        if predictor is None:
            self.coord_mapping = functools.partial(_default_coord_mapping,
                                                   self.search_range)
        else:
            self.coord_mapping = _wrap_predictor(self.search_range,
                                                 predictor, t)
        self._clean = False

    @property
    def kdtree(self):
        if not self._clean:
            self.rebuild()
        return self._kdtree

    def rebuild(self):
        """Rebuilds tree from ``points`` attribute.

        coord_map : function, optional

            Called with a list of N Point instances, returns their
            "effective" locations, as an N x d array (or list of tuples).
            Used for prediction (see "predict" module).

        rebuild() needs to be called after ``add_point()`` and
        before tree is used for spatial queries again (i.e. when
        memory is turned on).
        """
        self._clean = False
        if len(self.points) == 0:
            self._kdtree = None
        else:
            coords_mapped = self.coord_mapping(self.points)
            self._kdtree = cKDTree(coords_mapped, 15)
        # This could be tuned
        self._clean = True

    @property
    def coords(self):
        return _points_to_arr(self.points)

    @property
    def coords_mapped(self):
        if not self._clean:
            self.rebuild()
        if self._kdtree is None:
            return np.empty((0, self.ndim))
        else:
            return self._kdtree.data

    @property
    def coords_df(self):
        coords = self.coords
        if len(coords) == 0:
            return
        data = pd.DataFrame(coords, columns=default_pos_columns(self.ndim),
                            index=[p.uuid for p in self.points])

        # add placeholders to obtain columns with integer dtype
        data['frame'] = -1
        data['particle'] = -1
        for p in self.points:
            data.loc[p.uuid, 'frame'] = p.t
            data.loc[p.uuid, 'particle'] = p.track.id
            for col in p.extra_data:
                data.loc[p.uuid, col] = p.extra_data[col]
        return data

    def query_points(self, pos, max_dist_normed=1.):
        if self.kdtree is None:
            return
        pos_norm = pos / self.search_range
        found = self.kdtree.query_ball_point(pos_norm, max_dist_normed)
        found = set([i for sl in found for i in sl])  # ravel
        if len(found) == 0:
            return
        else:
            return self.coords[list(found)]


def assign_subnet(source, dest, subnets):
    """ Assign source point and dest point to the same subnet """
    i1 = source.subnet
    i2 = dest.subnet
    if i1 is None and i2 is None:
        raise ValueError("No subnet for added destination particle")
    if i1 == i2:  # if a and b are already in the same subnet, do nothing
        return
    if i1 is None:  # source did not belong to a subset before
        # just add it
        subnets[i2][0].add(source)
        source.subnet = i2
    elif i2 is None:  # dest did not belong to a subset before
        # just add it
        subnets[i1][1].add(dest)
        dest.subnet = i1
    else:  # source belongs to subset i1 before
        # merge the subnets
        subnets[i2][0].update(subnets[i1][0])
        subnets[i2][1].update(subnets[i1][1])
        # update the subnet identifiers per point
        for p in itertools.chain(*subnets[i1]):
            p.subnet = i2
        # and delete the old source subnet
        del subnets[i1]


def split_subnet(source, dest, new_range):
    # Clear the subnets and candidates for all points in both frames
    subnets = dict()
    for sp in source:
        sp.subnet = None
    for i, dp in enumerate(dest):
        dp.subnet = i
        subnets[i] = set(), {dp}

    for sp in source:
        for dp, dist in sp.forward_cands:
            if dist > new_range:
                continue
            assign_subnet(sp, dp, subnets=subnets)
    return (subnets[key] for key in subnets)


class Subnets(object):
    """ Class that evaluates the possible links between two groups of features.

    Candidates and subnet indices are stored inside the Point objects that are
    inside the provided TreeFinder objects.

    Subnets are based on the destination points: subnets having only a source
    point are not included. They can be accessed from the `lost` method.
    If subnets with only one source point need to be included, call the
    method `include_lost`. In that case, the method `lost` will raise.

    Parameters
    ----------
    source_hash : TreeFinder object
        The hash of the first (source) frame
    dest_hash : TreeFinder object
        The hash of the second (destination) frame
    max_neighbors : int, optional
        The maximum number of linking candidates for one feature. Default 10.

    Attributes
    ----------
    subnets : dictionary
        A dictonary, indexed by subnet index, that contains the subnets as a
        tuple of sets. The first set contains the source points, the second
        set contains the destination points. Iterate over this dictionary by
        directly iterating over the Subnets object.

    Methods
    -------
    get_lost :
        Lists source points without linking candidates ('lost' features).
        Raises if these particles are included in the subnets already, by
        calling `include_lost`.
    """
    def __init__(self, source_hash, dest_hash, max_neighbors=10):
        self.max_neighbors = max_neighbors
        self.source_hash = source_hash
        self.dest_hash = dest_hash
        self.includes_lost = False
        self.reset()
        self.compute()

    def reset(self):
        """ Clear the subnets and candidates for all points in both frames """
        self.subnets = dict()
        for p in self.source_hash.points:
            p.forward_cands = []
            p.subnet = None
        for i, p in enumerate(self.dest_hash.points):
            # p.back_cands = []
            p.subnet = i
            self.subnets[i] = set(), {p}

    def compute(self, search_range=1.):
        """ Evaluate the linking candidates and corresponding subnets, using
        given `search_range` (rescaled to 1.)."""
        source_hash = self.source_hash
        dest_hash = self.dest_hash
        if len(source_hash.points) == 0 or len(dest_hash.points) == 0:
            return
        search_range = float(search_range) + 1e-7
        dists, inds = source_hash.kdtree.query(dest_hash.coords_mapped,
                                               self.max_neighbors,
                                               distance_upper_bound=search_range)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, p in enumerate(dest_hash.points):
            for j in range(nn[i]):
                wp = source_hash.points[inds[i, j]]
                # p.back_cands.append((wp, dists[i, j]))
                wp.forward_cands.append((p, dists[i, j]))
                assign_subnet(wp, p, self.subnets)

    def add_dest_points(self, source_points, dest_points):
        """ Add destination points, evaluate candidates and subnets.

        This code cannot generate new subnets. The given points have to be such
        that new subnets do not have to be created.

        Parameters
        ----------
        source_points : iterable of points
            Consider these points only as linking candidates. They should exist
            already in Subnets.source_points.
        dest_points : iterable of points
            The destination points to add. They should be new.
        """
        # TODO is kdtree really faster here than brute force ?
        if len(dest_points) == 0:
            return
        source_points = list(source_points)
        source_coord = self.source_hash.coord_mapping(source_points)
        new_dest_hash = TreeFinder(dest_points, self.dest_hash.search_range)
        dists, inds = new_dest_hash.kdtree.query(source_coord,
                                                 max(len(source_points), 2),
                                                 distance_upper_bound=1+1e-7)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, source in enumerate(source_points):
            for j in range(nn[i]):
                dest = new_dest_hash.points[inds[i, j]]
                # dest.back_cands.append((source, dists[i, j]))
                source.forward_cands.append((dest, dists[i, j]))
                # source particle always has a subnet, add the dest particle
                self.subnets[source.subnet][1].add(dest)
                dest.subnet = source.subnet

        # sort candidates again because they might have changed
        for p in source_points:
            p.forward_cands.sort(key=lambda x: x[1])
        # for p in dest_hash.points:
        #    p.back_cands.sort(key=lambda x: x[1])

    def include_lost(self):
        """ Add source particles without any destination particle to the
        subnets."""
        if len(self.subnets) > 0:
            counter = itertools.count(start=max(self.subnets) + 1)
        else:
            counter = itertools.count()
        for p in self.source_hash.points:
            if len(p.forward_cands) == 0:
                subnet = next(counter)
                self.subnets[subnet] = {p}, set()
                p.subnet = subnet

        self.includes_lost = True

    def merge_lost_subnets(self):
        """ Merge subnets that have lost features and that are closer than
        twice the search range together, in order to account for the possibility
        that relocated points will join subnets together. """
        if not self.includes_lost:
            self.include_lost()

        # list subnets that have lost particles
        lost_source = []
        for key in self.subnets:
            source, dest = self.subnets[key]
            shortage = len(source) - len(dest)
            if shortage > 0:
                lost_source.extend(source)

        if len(lost_source) == 0:
            return
        lost_coords = self.source_hash.coord_mapping(lost_source)
        dists, inds = self.source_hash.kdtree.query(lost_coords, self.max_neighbors,
                                                    distance_upper_bound=2+1e-7)
        nn = np.sum(np.isfinite(dists), 1)  # Number of neighbors of each particle
        for i, p in enumerate(lost_source):
            for j in range(nn[i]):
                wp = self.source_hash.points[inds[i, j]]
                i1, i2 = p.subnet, wp.subnet
                if i1 != i2:
                    if i2 > i1:
                        i1, i2 = i2, i1
                    self.subnets[i2][0].update(self.subnets[i1][0])
                    self.subnets[i2][1].update(self.subnets[i1][1])
                    # update the subnet identifiers per point
                    for p in itertools.chain(*self.subnets[i1]):
                        p.subnet = i2
                    # and delete the old source subnet
                    del self.subnets[i1]

    def __iter__(self):
        return (self.subnets[key] for key in self.subnets)

    def lost(self):
        if self.includes_lost:
            raise ValueError('Lost particles are included in the subnets.')
        else:
            return [p for p in self.source_hash.points if p.subnet is None]


def subnet_linker_recursive(source_set, dest_set, search_range, **kwargs):
    if len(source_set) == 0 and len(dest_set) == 1:
        # no backwards candidates: particle will get a new track
        return [None], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 1:
        # one backwards candidate and one forward candidate
        return [source_set.pop()], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 0:
        # particle is lost. Not possible with default Linker implementation.
        return [source_set.pop()], [None]

    # sort candidates and add in penalty for not linking
    for _s in source_set:
        _s.forward_cands.sort(key=lambda x: x[1])
        _s.forward_cands.append((None, search_range))

    snl = SubnetLinker(source_set, len(dest_set), search_range, **kwargs)
    sn_spl, sn_dpl = [list(particles) for particles in zip(*snl.best_pairs)]

    for dp in dest_set - set(sn_dpl):
        # Unclaimed destination particle in subnet
        sn_spl.append(None)
        sn_dpl.append(dp)

    return sn_spl, sn_dpl


def subnet_linker_nonrecursive(source_set, dest_set, search_range, **kwargs):
    if len(source_set) == 0 and len(dest_set) == 1:
        # no backwards candidates: particle will get a new track
        return [None], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 1:
        # one backwards candidate and one forward candidate
        return [source_set.pop()], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 0:
        # particle is lost. Not possible with default Linker implementation.
        return [source_set.pop()], [None]

    # sort candidates and add in penalty for not linking
    for _s in source_set:
        _s.forward_cands.sort(key=lambda x: x[1])
        _s.forward_cands.append((None, search_range))

    sn_spl, sn_dpl = nonrecursive_link(source_set, len(dest_set), search_range, **kwargs)

    for dp in dest_set - set(sn_dpl):
        # Unclaimed destination particle in subnet
        sn_spl.append(None)
        sn_dpl.append(dp)

    return sn_spl, sn_dpl


def subnet_linker_numba(source_set, dest_set, search_range, **kwargs):
    if len(source_set) == 0 and len(dest_set) == 1:
        # no backwards candidates: particle will get a new track
        return [None], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 1:
        # one backwards candidate and one forward candidate
        return [source_set.pop()], [dest_set.pop()]
    elif len(source_set) == 1 and len(dest_set) == 0:
        # particle is lost. Not possible with default Linker implementation.
        return [source_set.pop()], [None]

    # sort candidates and add in penalty for not linking
    for _s in source_set:
        _s.forward_cands.sort(key=lambda x: x[1])
        _s.forward_cands.append((None, search_range))

    sn_spl, sn_dpl = numba_link(source_set, len(dest_set), search_range, **kwargs)

    for dp in dest_set - set(sn_dpl):
        # Unclaimed destination particle in subnet
        sn_spl.append(None)
        sn_dpl.append(dp)

    return sn_spl, sn_dpl


def adaptive_link_wrap(source_set, dest_set, search_range, subnet_linker,
                       adaptive_stop=None, adaptive_step=0.95, **kwargs):
    """Wraps a subnetlinker, making it adaptive."""
    try:
        sn_spl, sn_dpl = subnet_linker(source_set, dest_set,
                                       search_range, **kwargs)
    except SubnetOversizeException:
        if adaptive_stop is None:
            raise
        new_range = search_range * adaptive_step
        if search_range <= adaptive_stop:
            # adaptive_stop is the search_range below which linking
            # is presumed invalid. So we just give up.
            raise

        # Split the subnet and recurse
        sn_spl = []
        sn_dpl = []
        for source, dest in split_subnet(source_set, dest_set, new_range):
            split_spl, split_dpl = \
                adaptive_link_wrap(source, dest, new_range, subnet_linker,
                                   adaptive_stop, adaptive_step, **kwargs)
            sn_spl.extend(split_spl)
            sn_dpl.extend(split_dpl)

    return sn_spl, sn_dpl


def _sort_key_spl_dpl(x):
    if x[0] is not None:
        return list(x[0].pos)
    else:
        return list(x[1].pos)


def _points_from_arr(coords, frame_no, extra_data=None):
    """ Convert an ndarray of coordinates to a list of PointFindLink """
    if extra_data is None:
        return [PointFindLink(frame_no, pos) for pos in coords]
    else:
        return [PointFindLink(frame_no, pos,
                              extra_data={key: extra_data[key][i]
                                          for key in extra_data})
                for i, pos in enumerate(coords)]


class Linker(object):
    """ Re-implementation of trackpy.linking.Linker for use in find_link.

    Attributes
    ----------
    hash : TreeFinder
        The hash containing the points of the current level
    mem_set : set of points
    mem_history : list of sets of points
    particle_ids : list
        a list of track ids of the current hash
    points : list of points
        The points of the current hash.
    coords : ndarray
        The coordinates of the points of the current hash. It is possible to
        write on this attribute if the number of coordinates stays constant.
    coords_df : DataFrame
        The coordinates of the points of the current hash. It is possible to
        write on this attribute, changing positional coordinates only, if the
        number of coordinates stays constant.
    subnets : Subnets
        Subnets object containing the subnets of the prev and current points.

    Methods
    -------
    init_level(coords, t, extra_data)
        creates the first level (frame): no linking is done
    next_level(coords, t, extra_data)
        Add a level, assign candidates and subnets, and apply the links.
    update_hash(coords, t, extra_data)
        Updates the hash: the previous hash is returned
    assign_links()
        Assign links between previous and current points (given by obj.subnets)
        Returns a list of source particles and a list of destination particles
        that are to be linked.
    apply_links(spl, dpl)
        Applies links between the source particle list (spl) and destination
        particle list (dpl)
    """
    # Largest subnet we will attempt to solve.
    MAX_SUB_NET_SIZE = 30
    # For adaptive search, subnet linking should fail much faster.
    MAX_SUB_NET_SIZE_ADAPTIVE = 15
    # Maximum number of candidates per particle
    MAX_NEIGHBORS = 10

    def __init__(self, search_range, memory=0, subnet_linker=None,
                 predictor=None, adaptive_stop=None, adaptive_step=0.95):
        self.memory = memory
        self.predictor = predictor
        self.track_cls = TrackUnstored
        self.adaptive_stop = adaptive_stop
        self.adaptive_step = adaptive_step

        if subnet_linker is None or subnet_linker == 'auto':
            if NUMBA_AVAILABLE:
                subnet_linker = 'numba'
            else:
                subnet_linker = 'recursive'

        if subnet_linker == 'recursive':
            subnet_linker_func = subnet_linker_recursive
        elif subnet_linker == 'numba':
            subnet_linker_func = subnet_linker_numba
        elif subnet_linker == 'nonrecursive':
            subnet_linker_func = subnet_linker_nonrecursive
        elif callable(subnet_linker):
            subnet_linker_func = subnet_linker
        else:
            raise ValueError("Unknown subnet linker '{}'".format(subnet_linker))

        self.ndim = len(search_range)
        self.search_range = np.array(search_range)
        self.hash = None
        self.mem_set = set()

        if self.adaptive_stop is not None:
            # internal adaptive_stop is a fraction of search range
            adaptive_stop = np.max(adaptive_stop / self.search_range)
            if 1 * self.adaptive_stop <= 0:
                raise ValueError("adaptive_stop must be positive.")
            self.subnet_linker = functools.partial(adaptive_link_wrap,
                                                   subnet_linker=subnet_linker_func,
                                                   adaptive_stop=adaptive_stop,
                                                   adaptive_step=adaptive_step,
                                                   max_size=self.MAX_SUB_NET_SIZE_ADAPTIVE)
        else:
            self.subnet_linker = functools.partial(subnet_linker_func,
                                                   max_size=self.MAX_SUB_NET_SIZE)

    def update_hash(self, coords, t, extra_data=None):
        prev_hash = self.hash
        # add memory points to prev_hash (to be used as the next source)
        for m in self.mem_set:
            # add points to the hash
            prev_hash.add_point(m)
            # Record how many times this particle got "held back".
            # Since this particle has already been yielded in a previous
            # level, we can't store it there. We'll have to put it in the
            # track object, then copy this info to the point in cur_hash
            # if/when we make a link.
            m.track.incr_memory()
            # re-create the forward_cands list
            m.forward_cands = []

        # If prediction is enabled, we need to update the positions in prev_hash
        # to where we think they'll be in the frame corresponding to 'coords'.
        if prev_hash is not None and self.predictor is not None:
            prev_hash.set_predictor(self.predictor, t)  # Rewrite positions

        self.hash = TreeFinder(_points_from_arr(coords, t, extra_data),
                               self.search_range)
        return prev_hash

    def init_level(self, coords, t, extra_data=None):
        PointFindLink.reset_counter()
        TrackUnstored.reset_counter()
        self.mem_set = set()
        # Initialize memory with empty sets.
        self.mem_history = []
        for j in range(self.memory):
            self.mem_history.append(set())

        self.update_hash(coords, t, extra_data)
        # Assume everything in first level starts a Track.
        # Iterate over prev_level, not prev_set, because order -> track ID.
        for p in self.hash.points:
            TrackUnstored(p)

    @property
    def particle_ids(self):
        return [p.track.id for p in self.hash.points]

    @property
    def coords(self):
        return self.hash.coords
    @coords.setter
    def coords(self, value):
        if len(value) != len(self.hash.points):
            raise ValueError("Number of coordinates has changed")
        for coord, pnt in zip(value, self.hash.points):
            pnt.pos = coord
        self.hash._clean = False

    @property
    def coords_df(self):
        return self.hash.coords_df
    @coords_df.setter
    def coords_df(self, value):
        if len(value) != len(self.hash.points):
            raise ValueError("Number of features has changed")
        self.coords = value[default_pos_columns(self.ndim)].values

    def next_level(self, coords, t, extra_data=None):
        prev_hash = self.update_hash(coords, t, extra_data)

        self.subnets = Subnets(prev_hash, self.hash, self.MAX_NEIGHBORS)
        spl, dpl = self.assign_links()
        self.apply_links(spl, dpl)

    def assign_links(self):
        spl, dpl = [], []
        for source_set, dest_set in self.subnets:
            sn_spl, sn_dpl = self.subnet_linker(source_set, dest_set, 1.)
            spl.extend(sn_spl)
            dpl.extend(sn_dpl)

        # Leftovers
        lost = self.subnets.lost()
        spl.extend(lost)
        dpl.extend([None] * len(lost))

        return spl, dpl

    def apply_links(self, spl, dpl):
        new_mem_set = set()
        for sp, dp in sorted(zip(spl, dpl), key=_sort_key_spl_dpl):
            # Do linking
            if sp is not None and dp is not None:
                sp.track.add_point(dp)
                if sp in self.mem_set:  # Very rare
                    self.mem_set.remove(sp)
            elif sp is None:
                # if unclaimed destination particle, a track is born!
                TrackUnstored(dp)
            elif dp is None:
                # add the unmatched source particles to the new
                # memory set
                new_mem_set.add(sp)

            # # Clean up
            # if dp is not None:
            #     dp.back_cands = []
            if sp is not None:
                sp.forward_cands = []

        # add in the memory points
        # store the current level for use in next loop
        if self.memory > 0:
            # identify the new memory points
            new_mem_set -= self.mem_set
            self.mem_history.append(new_mem_set)
            # remove points that are now too old
            self.mem_set -= self.mem_history.pop(0)
            # add the new points
            self.mem_set |= new_mem_set


class FindLinker(Linker):
    """ Linker that relocates lost features.

    Newly found features are farther than ``separation`` from any other feature
    in the current frame, closer than ``search_range`` to a feature in the
    previous frame, and have minimum integrated intensity ``minmass`` in the
    feature region (defined by ``diameter``).

    Parameters
    ----------
    search_range : tuple
        The maximum distance features can move between frames, in pixels.
    separation : tuple
        The minimum distance between features, in pixels.
    diameter : tuple
        Size used in the characterization of new features.
        Also determines the margin (margin = diameter // 2).
    memory : int, optional
        Default 0
    minmass : number, optional
        Minimum summed intensity (in the masked image) of relocated features.
        Default 0.
    percentile : number, optional
        Precentile threshold used in local maxima finding. Default 64.


    Methods
    -------
    next_level(coords, t, image, extra_data)
        Link and relocate the next frame, using the extra parameter ``image``.
    relocate(source_points, n)
        Relocate ``n`` points close to source_points
    get_relocate_candidates(source_points)
        Obtain relacote coordinates of new features close to ``source_points``

    See also
    --------
    Linker
    """
    def __init__(self, search_range, separation, diameter=None,
                 minmass=0, percentile=64, **kwargs):
        super(FindLinker, self).__init__(search_range, **kwargs)
        if diameter is None:
            diameter = separation
        self.radius = tuple([int(d // 2) for d in diameter])
        self.separation = separation
        self.minmass = minmass  # in masked image
        self.percentile = percentile

        # For grey dilation: find the largest box that fits inside the ellipse
        # given by separation
        self.dilation_size = tuple([int(2 * s / np.sqrt(self.ndim))
                                   for s in self.separation])
        # slice_radius: radius for relocate mask
        # search_range + feature radius + 1
        self.slice_radius = tuple([int(s + r + 1)
                                   for (s, r) in zip(self.search_range,
                                                     self.radius)])
        # background_radius: radius to make sure the already located features
        # do not fall inside slice radius
        bg_radius = [sl + r + 1 for (sl, r) in zip(self.slice_radius,
                                                   self.radius)]
        # The big feature hashtable is normed to search_range. For performance,
        # we do not rebuild this large hashtable. apply the norm here and take
        # the largest value.
        self.bg_radius = max([a / b for (a, b) in zip(bg_radius,
                                                      self.search_range)])
        self.threshold = (None, None)

    def next_level(self, coords, t, image, extra_data=None):
        self.image = image
        self.curr_t = t
        super(FindLinker, self).next_level(coords, t, extra_data)

    def relocate(self, pos, n=1):
        candidates, extra_data = self.get_relocate_candidates(pos)
        if candidates is None:
            return set()
        else:
            n = min(n, len(candidates))
            points = _points_from_arr(candidates[:n], self.curr_t,
                                      extra_data=extra_data)
        return set(points)

    def percentile_threshold(self, percentile):
        frame_no, threshold = self.threshold
        if self.curr_t != frame_no:
            not_black = self.image[np.nonzero(self.image)]
            if len(not_black) == 0:
                threshold = None
            else:
                threshold = np.percentile(not_black, percentile)
            self.threshold = (self.curr_t, threshold)
        return threshold

    def get_relocate_candidates(self, pos):
        # pos are the estimated locations of the features (ndarray N x ndim)
        pos = np.atleast_2d(pos)

        # slice region around cluster
        im_unmasked, origin = slice_image(pos, self.image, self.slice_radius)

        # return when there is no intensity left
        if im_unmasked.sum() == 0:
            return None, None
        # mask image so that only regions up to slice_radius are visible
        im_masked = mask_image(pos, im_unmasked, self.slice_radius, origin,
                               invert=False)
        # return when there is no intensity left
        if im_masked.sum() == 0:
            return None, None

        # mask coords that were already found ('background')
        background = self.hash.query_points(pos, self.bg_radius)
        if background is not None:
            im_masked = mask_image(background, im_masked, self.separation,
                                   origin, invert=True)

        threshold = self.percentile_threshold(self.percentile)
        if threshold is None:  # completely black image
            return None, None
        if np.all(im_masked < threshold):  # image entirely below threshold
            return None, None
        # The intersection of the image with its dilation gives local maxima.
        dilation = ndimage.grey_dilation(im_masked, self.dilation_size,
                                         mode='constant')
        maxima = (im_masked == dilation) & (im_masked > threshold)
        if np.sum(maxima) == 0:   # no maxima
            return None, None
        coords = np.vstack(np.where(maxima)).T

        # Do not accept peaks near the edges.
        shape = np.array(self.image.shape)
        near_edge = np.any((coords < self.radius) |
                           (coords > (shape - self.radius - 1)), axis=1)
        coords = coords[~near_edge]
        if len(coords) == 0:
            return None, None

        # drop points that are further than search range from any initial point
        max_dist = np.atleast_2d(self.search_range)
        kdtree = cKDTree(coords / max_dist, 30)
        found = kdtree.query_ball_point((pos - origin) / max_dist, 1.)
        if len(found) > 0:
            coords = coords[list(set([i for sl in found for i in sl]))]
        else:
            return None, None

        # drop dimmer points that are closer than separation to each other
        coords = drop_close(coords, self.separation,
                            [im_masked[tuple(c)] for c in coords])
        if coords is None:
            return None, None

        try:
            scale_factor = self.image.metadata['scale_factor']
        except (AttributeError, KeyError):
            scale_factor = 1.
        extra_data = characterize(coords, im_masked, self.radius, scale_factor)

        mass = extra_data['mass']
        mask = np.argsort(mass)[::-1][:np.sum(mass >= self.minmass)]
        for key in extra_data:
            extra_data[key] = extra_data[key][mask]
        return coords[mask] + origin, extra_data

    def assign_links(self):
        # The following method includes subnets with only one source point
        self.subnets.include_lost()
        # Also, it merges subnets that are less than 2*search_range spaced,
        # to account for lost particles that link subnets together. A possible
        # performance enhancement would be joining subnets together during
        # iterating over the subnets.
        self.subnets.merge_lost_subnets()

        spl, dpl = [], []
        for source_set, dest_set in self.subnets:
            # relocate if necessary
            shortage = len(source_set) - len(dest_set)
            if shortage > 0:
                if self.predictor is not None:
                    # lookup the predicted locations
                    sh = self.subnets.source_hash
                    pos = [c for c, p in zip(sh.coords_mapped,
                                             sh.points) if p in source_set]
                else:
                    pos = [s.pos for s in source_set]
                new_cands = self.relocate(pos, shortage)
                # this adapts the dest_set inplace
                self.subnets.add_dest_points(source_set, new_cands)
            else:
                new_cands = set()

            # link
            sn_spl, sn_dpl = self.subnet_linker(source_set, dest_set, 1.)

            # list the claimed destination particles and add them to the hash
            sn_dpl_set = set(sn_dpl)
            # claimed new destination particles
            for p in new_cands & sn_dpl_set:
                self.hash.add_point(p)
            # unclaimed old destination particles
            unclaimed = (dest_set - sn_dpl_set) - new_cands
            sn_spl.extend([None] * len(unclaimed))
            sn_dpl.extend(unclaimed)

            spl.extend(sn_spl)
            dpl.extend(sn_dpl)

        return spl, dpl
