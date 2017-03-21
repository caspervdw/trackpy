from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import itertools
import numpy as np
import pandas as pd
from pandas import DataFrame
import nose
from numpy.testing import assert_equal

from trackpy.utils import pandas_sort
from trackpy.artificial import CoordinateReader
from trackpy.find_link import find_link
from trackpy.tests.common import assert_traj_equal, StrictTestCase
from trackpy.tests.test_link_new import SubnetNeededTests


class FindLinkTests(SubnetNeededTests):
    def setUp(self):
        super(FindLinkTests, self).setUp()
        self.linker_opts['separation'] = 10
        self.linker_opts['diameter'] = 15

    def link(self, f, search_range, *args, **kwargs):
        # the minimal spacing between features in f is assumed to be 1.

        # from scipy.spatial import cKDTree
        # mindist = 1e7
        # for _, _f in f.groupby('frame'):
        #     dists, _ = cKDTree(_f[['y', 'x']].values).query(_f[['y', 'x']].values, k=2)
        #     mindist = min(mindist, dists[:, 1].min())
        # print("Minimal dist is {0:.3f}".format(mindist))

        kwargs = dict(self.linker_opts, **kwargs)
        size = 3
        separation = kwargs['separation']
        f = f.copy()
        f[['y', 'x']] *= separation
        topleft = (f[['y', 'x']].min().values - 4 * separation).astype(np.int)
        f[['y', 'x']] -= topleft
        shape = (f[['y', 'x']].max().values + 4 * separation).astype(np.int)
        reader = CoordinateReader(f, shape, size)
        if kwargs.get('adaptive_stop', None) is not None:
            kwargs['adaptive_stop'] *= separation
        result = find_link(reader,
                           search_range=search_range*separation,
                           *args, **kwargs)
        result = pandas_sort(result, ['particle', 'frame']).reset_index(drop=True)
        result[['y', 'x']] += topleft
        result[['y', 'x']] /= separation
        return result

    def test_args_dtype(self):
        """Check whether find_link accepts float typed arguments"""
        # One 1D stepper
        N = 5
        f = DataFrame(
            {'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)

        # Should not raise
        actual = self.link(f, 5.2, separation=9.5, diameter=15.2)
        assert_traj_equal(actual, expected)


class FindLinkOneFailedFindTests(FindLinkTests):
    def setUp(self):
        super(FindLinkOneFailedFindTests, self).setUp()
        FAIL_FRAME = dict(test_isolated_continuous_random_walks=5,
                          test_nearby_continuous_random_walks=10,
                          test_start_at_frame_other_than_zero=4,
                          test_two_nearby_steppers_one_gapped=2,
                          test_two_isolated_steppers_one_gapped=2)

        test_name = self.id()[self.id().rfind('.') + 1:]
        fail_frame = FAIL_FRAME.get(test_name, 3)

        def callback(image, coords, **unused_kwargs):
            if image.frame_no == fail_frame:
                return np.empty((0, 2))
            else:
                return coords
        self.linker_opts['before_link'] = callback


class FindLinkManyFailedFindTests(FindLinkTests):
    def setUp(self):
        super(FindLinkManyFailedFindTests, self).setUp()
        FAIL_FRAME = dict(test_isolated_continuous_random_walks=5,
                          test_nearby_continuous_random_walks=10,
                          test_start_at_frame_other_than_zero=4,
                          test_two_nearby_steppers_one_gapped=5,
                          test_two_isolated_steppers_one_gapped=5,
                          test_blank_frame_no_memory=5)

        test_name = self.id()[self.id().rfind('.') + 1:]
        fail_frame = FAIL_FRAME.get(test_name, 3)

        def callback(image, coords, **unused_kwargs):
            if image.frame_no >= fail_frame:
                return np.empty((0, 2))
            else:
                return coords
        self.linker_opts['before_link'] = callback


class FindLinkSpecialCases(StrictTestCase):
    # also, for paper images
    do_diagnostics = False  # Don't ask for diagnostic info from linker
    def setUp(self):
        self.linker_opts = dict()
        self.search_range = 12
        self.separation = 4
        self.diameter = 12  # only for characterization
        self.size = 3

    def link(self, f, shape, remove=None, **kwargs):
        _kwargs = dict(diameter=self.diameter,
                       search_range=self.search_range,
                       separation=self.separation)
        _kwargs.update(kwargs)
        if remove is not None:
            callback_coords = f.loc[f['frame'] == 1, ['y', 'x']].values
            remove = np.array(remove, dtype=np.int)
            if np.any(remove < 0) or np.any(remove > len(callback_coords)):
                raise RuntimeError('Invalid test: `remove` is out of bounds.')
            callback_coords = np.delete(callback_coords, remove, axis=0)
            def callback(image, coords, **unused_kwargs):
                if image.frame_no == 1:
                    return callback_coords
                else:
                    return coords
        else:
            callback = None
        reader = CoordinateReader(f, shape, self.size)
        return find_link(reader, before_link=callback, **_kwargs)

    def test_in_margin(self):
        expected = DataFrame({'x': [12, 6], 'y': [12, 5],
                              'frame': [0, 1], 'particle': [0, 0]})

        actual = self.link(expected, shape=(24, 24))
        assert_equal(len(actual), 1)

    def test_one(self):
        expected = DataFrame({'x': [8, 16], 'y': [16, 16],
                              'frame': [0, 1], 'particle': [0, 0]})

        actual = self.link(expected, shape=(24, 24), remove=[0])
        assert_traj_equal(actual, expected)
        actual = self.link(expected, shape=(24, 24), search_range=7, remove=[0])
        assert_equal(len(actual), 1)

    def test_two_isolated(self):
        shape = (32, 32)
        expected = DataFrame({'x': [8, 16, 24, 16], 'y': [8, 8, 24, 24],
                                 'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_two_single_overlap(self):
        shape = (16, 40)
        # a --> b  c --> d    : b-c overlap
        expected = DataFrame({'x': [8, 16, 24, 32], 'y': [8, 8, 8, 8],
                              'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # a --> b  d <-- c    : b-d overlap
        expected = DataFrame({'x': [8, 16, 24, 32], 'y': [8, 8, 8, 8],
                              'frame': [0, 1, 1, 0], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # b <-- a  d --> c    : a-d overlap
        expected = DataFrame({'x': [8, 16, 24, 32], 'y': [8, 8, 8, 8],
                              'frame': [1, 0, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_two_double_overlap(self):
        shape = (24, 32)
        # (a b) c --> d    # a-c and b-c overlap
        expected = DataFrame({'x': [8, 8, 16, 24], 'y': [8, 16, 12, 12],
                              'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # (a b) d <-- c    # a-d and b-d overlap
        expected = DataFrame({'x': [8, 8, 16, 24], 'y': [8, 16, 12, 12],
                              'frame': [0, 1, 1, 0], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_two_triple_overlap(self):
        shape = (24, 32)
        # a --> b
        #     c --> d    # a-c, b-c, and b-d overlap
        expected = DataFrame({'x': [8, 16, 16, 24], 'y': [8, 8, 16, 16],
                              'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # a --> b
        #     d <-- c    # a-d, b-d, and b-c overlap
        expected = DataFrame({'x': [8, 16, 16, 24], 'y': [8, 8, 16, 16],
                              'frame': [0, 1, 1, 0], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)
        # b <-- a
        #     c -- > d    # b-c, a-c, and a-d overlap
        expected = DataFrame({'x': [8, 16, 16, 24], 'y': [8, 8, 16, 16],
                              'frame': [1, 0, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_two_full_overlap(self):
        shape = (24, 24)
        # a --> b
        # c --> d    # a-c, b-c, b-d, a-d overlap
        expected = DataFrame({'x': [8, 15, 8, 15], 'y': [8, 8, 16, 16],
                              'frame': [0, 1, 0, 1], 'particle': [0, 0, 1, 1]})
        for remove in [[], [0], [1], [0, 1]]:
            actual = self.link(expected, shape=shape, remove=remove)
            assert_traj_equal(actual, expected)

    def test_merging_subnets(self):
        shape = (24, 48)
        expected = pd.DataFrame({'x': [8, 12, 16, 20, 32, 28, 40, 36],
                                 'y': [8, 16, 8, 16, 8, 16, 8, 16],
                                 'frame': [0, 1, 0, 1, 0, 1, 0, 1],
                                 'particle': [0, 0, 1, 1, 2, 2, 3, 3]})

        for n in range(5):
            for remove in itertools.combinations(range(4), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)

    def test_splitting_subnets(self):
        shape = (24, 48)
        expected = pd.DataFrame({'x': [8, 12, 16, 20, 32, 28, 40, 36],
                                 'y': [8, 16, 8, 16, 8, 16, 8, 16],
                                 'frame': [1, 0, 1, 0, 1, 0, 1, 0],
                                 'particle': [0, 0, 1, 1, 2, 2, 3, 3]})

        for n in range(5):
            for remove in itertools.combinations(range(4), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)

    def test_shifting_string(self):
        shape = (24, 48)
        shift = 5
        expected = pd.DataFrame({'x': [8, 8+shift, 16, 16+shift,
                                       24, 24+shift, 32, 32+shift],
                                 'y': [8, 16, 8, 16, 8, 16, 8, 16],
                                 'frame': [0, 1, 0, 1, 0, 1, 0, 1],
                                 'particle': [0, 0, 1, 1, 2, 2, 3, 3]})

        for n in range(5):
            for remove in itertools.combinations(range(4), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)

    def test_multiple_lost_simple(self):
        shape = (32, 32)
        #   b      a, b, c, d in frame 0
        # a e c    e in frame 1, disappears, should be linked to correct one
        #   d
        # left
        expected = pd.DataFrame({'x': [8, 16, 24, 16, 15],
                                 'y': [16, 8, 16, 24, 16],
                                 'frame': [0, 0, 0, 0, 1],
                                 'particle': [0, 1, 2, 3, 0]})
        actual = self.link(expected, shape=shape, remove=[0])
        assert_traj_equal(actual, expected)
        # top
        expected = pd.DataFrame({'x': [8, 16, 24, 16, 16],
                                 'y': [16, 8, 16, 24, 15],
                                 'frame': [0, 0, 0, 0, 1],
                                 'particle': [0, 1, 2, 3, 1]})
        actual = self.link(expected, shape=shape, remove=[0])
        assert_traj_equal(actual, expected)
        # right
        expected = pd.DataFrame({'x': [8, 16, 24, 16, 17],
                                 'y': [16, 8, 16, 24, 16],
                                 'frame': [0, 0, 0, 0, 1],
                                 'particle': [0, 1, 2, 3, 2]})
        actual = self.link(expected, shape=shape, remove=[0])
        assert_traj_equal(actual, expected)
        # bottom
        expected = pd.DataFrame({'x': [8, 16, 24, 16, 16],
                                 'y': [16, 8, 16, 24, 17],
                                 'frame': [0, 0, 0, 0, 1],
                                 'particle': [0, 1, 2, 3, 3]})
        actual = self.link(expected, shape=shape, remove=[0])
        assert_traj_equal(actual, expected)

    def test_multiple_lost_subnet(self):
        shape = (24, 48)
        #  (subnet 1, a-c) g (subnet 2, d-f)
        # left
        expected = pd.DataFrame({'x': [8, 10, 16, 32, 40, 38, 23],
                                 'y': [8, 16, 8, 8, 8, 16, 8],
                                 'frame': [0, 1, 0, 0, 0, 1, 1],
                                 'particle': [0, 0, 1, 2, 3, 3, 1]})
        for n in range(3):
            for remove in itertools.combinations(range(3), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)
        # right
        expected = pd.DataFrame({'x': [8, 10, 16, 32, 40, 38, 25],
                                 'y': [8, 16, 8, 8, 8, 16, 8],
                                 'frame': [0, 1, 0, 0, 0, 1, 1],
                                 'particle': [0, 0, 1, 2, 3, 3, 2]})
        for n in range(3):
            for remove in itertools.combinations(range(3), n):
                actual = self.link(expected, shape=shape, remove=remove)
                assert_traj_equal(actual, expected)
