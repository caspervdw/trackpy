from __future__ import (absolute_import, division, print_function,
                        unicode_literals)
import six
import os
import numpy as np
import pandas as pd
from pandas import DataFrame
import nose
from numpy.testing import assert_equal

from trackpy.try_numba import NUMBA_AVAILABLE
from trackpy.linking import SubnetOversizeException
from trackpy.utils import pandas_sort
from trackpy.find_link import link_simple, link_simple_iter, link_simple_df_iter, verify_integrity
from trackpy.tests.common import assert_traj_equal, StrictTestCase
from trackpy.tests.test_link import random_walk

path, _ = os.path.split(os.path.abspath(__file__))
path = os.path.join(path, 'data')


def _skip_if_no_numba():
    if not NUMBA_AVAILABLE:
        raise nose.SkipTest('numba not installed. Skipping.')

def contracting_grid():
    """Two frames with a grid of 441 points.

    In the second frame, the points contract, so that the outermost set
    coincides with the second-outermost set in the previous frame.

    This is a way to challenge (and/or stump) a subnet solver.
    """
    pts0x, pts0y = np.mgrid[-10:11, -10:11] * 2.
    pts0 = pd.DataFrame(dict(x=pts0x.flatten(), y=pts0y.flatten(),
                             frame=0))
    pts1 = pts0.copy()
    pts1.frame = 1
    pts1.x = pts1.x * 0.9
    pts1.y = pts1.y * 0.9
    allpts = pd.concat([pts0, pts1], ignore_index=True)
    allpts.x += 200  # Because BTree doesn't allow negative coordinates
    allpts.y += 200
    return allpts


class CommonTrackingTests(StrictTestCase):
    def setUp(self):
        self.linker_opts = dict(link_strategy='recursive')

    def test_one_trivial_stepper(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

    def test_output_dtypes(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                       'frame': np.arange(N)})
        # Integer-typed input
        f['frame'] = f['frame'].astype(np.int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.int)

        # Float-typed input
        f['frame'] = f['frame'].astype(np.float)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.int)

    def test_two_isolated_steppers(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first, so the particle
        # labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_two_isolated_steppers_one_gapped(self):
        N = 5
        Y = 25
        # Begin second feature one frame later than the first,
        # so the particle labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': np.arange(N)})
        a = a.drop(3).reset_index(drop=True)
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1),
                      'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy()
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)
        # link_df_iter() tests not performed, because hash_size is
        # not knowable from the first frame alone.

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_isolated_continuous_random_walks(self):
        # Two 2D random walks
        np.random.seed(0)
        N = 30
        Y = 250
        M = 20 # margin, because negative values raise OutOfHash
        a = DataFrame({'x': M + random_walk(N), 'y': M + random_walk(N), 'frame': np.arange(N)})
        b = DataFrame({'x': M + random_walk(N - 1), 'y': M + Y + random_walk(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Many 2D random walks
        np.random.seed(0)
        initial_positions = [(100, 100), (200, 100), (100, 200), (200, 200)]
        import itertools
        c = itertools.count()
        def walk(x, y):
            i = next(c)
            return DataFrame({'x': x + random_walk(N - i),
                              'y': y + random_walk(N - i),
                             'frame': np.arange(i, N)})
        f = pd.concat([walk(*pos) for pos in initial_positions])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

    def test_start_at_frame_other_than_zero(self):
        # One 1D stepper
        N = 5
        FIRST_FRAME = 3
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': FIRST_FRAME + np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

    def test_blank_frame_no_memory(self):
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                      'frame': [0, 1, 2, 4, 5],
                      'particle': [0, 0, 0, 1, 1]})
        expected = f.copy()
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

    def test_real_data_that_causes_duplicate_bug(self):
        filename = 'reproduce_duplicate_track_assignment.df'
        f = pd.read_pickle(os.path.join(path, filename))
        # Not all parameters reproduce it, but these do
        f = self.link(f, 8, memory=2)
        verify_integrity(f)


    # def test_search_range(self):
    #     t = self.link(unit_steps(), 1.1, hash_generator((10, 10), 1))
    #     assert len(t) == 1  # One track
    #     t_short = self.link(unit_steps(), 0.9, hash_generator((10, 10), 1))
    #     assert len(t_short) == len(
    #         unit_steps())  # Each step is a separate track.
    #
    #     t = self.link(random_walk_legacy(), max_disp + 0.1,
    #                   hash_generator((10, 10), 1))
    #     assert len(t) == 1  # One track
    #     t_short = self.link(random_walk_legacy(), max_disp - 0.1,
    #                         hash_generator((10, 10), 1))
    #     assert len(t_short) > 1  # Multiple tracks
    #
    # def test_box_size(self):
    #     """No matter what the box size, there should be one track, and it should
    #     contain all the points."""
    #     for box_size in [0.1, 1, 10]:
    #         t1 = self.link(unit_steps(), 1.1,
    #                        hash_generator((10, 10), box_size))
    #         t2 = self.link(random_walk_legacy(), max_disp + 1,
    #                        hash_generator((10, 10), box_size))
    #         assert len(t1) == 1
    #         assert len(t2) == 1
    #         assert len(t1[0].points) == len(unit_steps())
    #         assert len(t2[0].points) == len(random_walk_legacy())
    #
    # def test_easy_tracking(self):
    #     level_count = 5
    #     p_count = 16
    #     levels = []
    #
    #     for j in range(level_count):
    #         level = []
    #         for k in np.arange(p_count) * 2:
    #             level.append(PointND(j, (j, k)))
    #         levels.append(level)
    #
    #     hash_generator = lambda: Hash_table((level_count + 1,
    #                                          p_count * 2 + 1), .5)
    #     tracks = self.link(levels, 1.5, hash_generator)
    #
    #     assert len(tracks) == p_count
    #
    #     for t in tracks:
    #         x, y = zip(*[p.pos for p in t])
    #         dx = np.diff(x)
    #         dy = np.diff(y)
    #
    #         assert np.sum(dx) == level_count - 1
    #         assert np.sum(dy) == 0

    def test_copy(self):
        """Check inplace/copy behavior of link_df """
        # One 1D stepper
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        expected = f.copy()
        expected['particle'] = np.zeros(N)

        # Should copy
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)
        assert 'particle' not in f.columns

    @nose.tools.raises(SubnetOversizeException)
    def test_oversize_fail(self):
        df = contracting_grid()
        self.link(df, search_range=2)

    @nose.tools.raises(SubnetOversizeException)
    def test_adaptive_fail(self):
        """Check recursion limit"""
        self.link(contracting_grid(), search_range=2, adaptive_stop=1.84)

    def link(self, f, search_range, *args, **kwargs):
        kwargs = dict(self.linker_opts, **kwargs)
        return link_simple(f, search_range, *args, **kwargs)


class SubnetNeededTests(CommonTrackingTests):
    def test_two_nearby_steppers(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle
        # labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_two_nearby_steppers_one_gapped(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle
        # labeling (0, 1) is established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 2]), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_nearby_continuous_random_walks(self):
        # Two 2D random walks
        np.random.seed(0)
        N = 30
        Y = 250
        M = 20 # margin, because negative values raise OutOfHash
        a = DataFrame({'x': M + random_walk(N),
                       'y': M + random_walk(N),
                       'frame': np.arange(N)})
        b = DataFrame({'x': M + random_walk(N - 1),
                       'y': M + Y + random_walk(N - 1),
                       'frame': np.arange(1, N)})
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.zeros(N), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        np.random.seed(0)
        initial_positions = [(10, 11), (10, 18), (14, 15), (20, 21), (13, 13),
                             (10, 10), (17, 19)]
        import itertools
        c = itertools.count()
        def walk(x, y):
            i = next(c)
            return DataFrame({'x': x + random_walk(N - i),
                              'y': y + random_walk(N - i),
                              'frame': np.arange(i, N)})
        f = pd.concat([walk(*pos) for pos in initial_positions])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([i*np.ones(N - i) for i in range(len(initial_positions))])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        actual = self.link(f, 5)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5)
        assert_traj_equal(actual, expected)

    def test_quadrature_distances(self):
        """A simple test to check whether the subnet linker adds
        distances in quadrature (as in Crocker-Grier)."""
        def subnet_test(epsilon):
            """Returns 2 features in 2 frames, which represent a special
            case when the subnet linker adds distances in quadrature. With
            epsilon=0, subnet linking is degenerate. Therefore
            linking should differ for positive and negative epsilon."""
            return pd.DataFrame([(0, 10, 11), (0, 10, 8),
                                 (1, 9, 10), (1, 12, 10 + epsilon)],
                                columns=['frame', 'x', 'y'])
        trneg = self.link(subnet_test(0.01), 5)
        trpos = self.link(subnet_test(-0.01), 5)
        assert not np.allclose(trneg.particle.values, trpos.particle.values)

    def test_penalty(self):
        """A test case of two particles, spaced 8 and each moving by 8 down
        and 7 to the right. We have two likely linking results:

        1. two links, total squared displacement = 2*(8**2 + 7**2) = 226
        2. one link, total squared displacement = (8**2 + 1**2) + sr**2

        Case 2 gets a penalty for not linking, which equals the search range
        squared. We vary this in this test.

        With a penalty of 13, case 2 has a total cost of 234 and we expect case
        1. as the result.

        With a penalty of 12, case 2. will have a total cost of 209 and we
        expect case 2. as the result.
        """
        f = pd.DataFrame({'x': [0, 8, 7, 8 + 7],
                          'y': [0, 0, 8, 8],
                          'frame': [0, 0, 1, 1]})
        case1 = f.copy()
        case1['particle'] = np.array([0, 1, 0, 1])
        case2 = f.copy()
        case2['particle'] = np.array([0, 1, 1, 2])

        actual = self.link(f, 13)
        pandas_sort(case1, ['x'], inplace=True)
        pandas_sort(actual, ['x'], inplace=True)
        assert_equal(actual['particle'].values.astype(np.int),
                     case1['particle'].values.astype(np.int))

        actual = self.link(f, 12)
        pandas_sort(case2, ['x'], inplace=True)
        pandas_sort(actual, ['x'], inplace=True)
        assert_equal(actual['particle'].values.astype(np.int),
                     case2['particle'].values.astype(np.int))

    # def test_memory(self):
    #     """A unit-stepping trajectory and a random walk are observed
    #     simultaneously. The random walk is missing from one observation."""
    #     a = [p[0] for p in unit_steps()]
    #     b = [p[0] for p in random_walk_legacy()]
    #     # b[2] is intentionally omitted below.
    #     gapped = lambda: deepcopy([[a[0], b[0]], [a[1], b[1]], [a[2]],
    #                                [a[3], b[3]], [a[4], b[4]]])
    #     safe_disp = 1 + random_x.max() - random_x.min()  # Definitely large enough
    #     t0 = self.link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=0)
    #     assert len(t0) == 3, len(t0)
    #     t2 = self.link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=2)
    #     assert len(t2) == 2, len(t2)
    #
    # def test_memory_removal(self):
    #     """BUG: A particle remains in memory after its Track is resumed, leaving two
    #     copies that can independently pick up desinations, leaving two Points in the
    #     same Track in a single level."""
    #     levels  = []
    #     levels.append([PointND(0, [1, 1]), PointND(0, [4, 1])])  # two points
    #     levels.append([PointND(1, [1, 1])])  # one vanishes, but is remembered
    #     levels.append([PointND(2, [1, 1]), PointND(2, [2, 1])]) # resume Track
    #     levels.append([PointND(3, [1, 1]), PointND(3, [2, 1]), PointND(3, [4, 1])])
    #     t = self.link(levels, 5, hash_generator((10, 10), 1), memory=2)
    #     assert len(t) == 3, len(t)
    #
    # def test_memory_with_late_appearance(self):
    #     a = [p[0] for p in unit_steps()]
    #     b = [p[0] for p in random_walk_legacy()]
    #     gapped = lambda: deepcopy([[a[0]], [a[1], b[1]], [a[2]],
    #                                [a[3]], [a[4], b[4]]])
    #     safe_disp = 1 + random_x.max() - random_x.min()  # large enough
    #     t0 = self.link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=1)
    #     assert len(t0) == 3, len(t0)
    #     t2 = self.link(gapped(), safe_disp, hash_generator((10, 10), 1), memory=4)
    #     assert len(t2) == 2, len(t2)

    def test_memory_on_one_gap(self):
        N = 5
        Y = 2
        # Begin second feature one frame later than the first, so the particle labeling (0, 1) is
        # established and not arbitrary.
        a = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        b = DataFrame({'x': np.arange(1, N), 'y': Y + np.ones(N - 1), 'frame': np.arange(1, N)})
        a = a.drop(3).reset_index(drop=True)
        f = pd.concat([a, b])
        expected = f.copy().reset_index(drop=True)
        expected['particle'] = np.concatenate([np.array([0, 0, 0, 0]), np.ones(N - 1)])
        pandas_sort(expected, ['particle', 'frame'], inplace=True)
        expected.reset_index(drop=True, inplace=True)
        actual = self.link(f, 5, memory=1)
        assert_traj_equal(actual, expected)

        # Sort rows by frame (normal use)
        actual = self.link(pandas_sort(f, 'frame'), 5, memory=1)
        assert_traj_equal(actual, expected)

        # Shuffle rows (crazy!)
        np.random.seed(0)
        f1 = f.reset_index(drop=True)
        f1.reindex(np.random.permutation(f1.index))
        actual = self.link(f1, 5, memory=1)
        assert_traj_equal(actual, expected)

    # def test_pathological_tracking(self):
    #     level_count = 5
    #     p_count = 16
    #     levels = []
    #     shift = 1
    #
    #     for j in range(level_count):
    #         level = []
    #         for k in np.arange(p_count) * 2:
    #             level.append(PointND(k // 2, (j, k + j * shift)))
    #         levels.append(level)
    #
    #     hash_generator = lambda: Hash_table((level_count + 1,
    #                                          p_count*2 + level_count*shift + 1),
    #                                         .5)
    #     tracks = self.link(levels, 8, hash_generator)
    #
    #     assert len(tracks) == p_count, len(tracks)

    def test_adaptive_range(self):
        """Tests that is unbearably slow without a fast subnet linker."""
        cg = contracting_grid()
        # Allow 5 applications of the step
        tracks = self.link(cg, 2, adaptive_step=0.8, adaptive_stop=0.64)
        # Transform back to origin
        tracks.x -= 200
        tracks.y -= 200
        assert len(cg) == len(tracks)
        tr0 = tracks[tracks.frame == 0].set_index('particle')
        tr1 = tracks[tracks.frame == 1].set_index('particle')
        only0 = list(set(tr0.index) - set(tr1.index))
        only1 = list(set(tr1.index) - set(tr0.index))
        # From the first frame, the outermost particles should have been lost.
        assert all(
            (tr0.x.ix[only0].abs() > 19) | (tr0.y.ix[only0].abs() > 19))
        # There should be new tracks in the second frame, corresponding to the
        # middle radii.
        assert all(
            (tr1.x.ix[only1].abs() == 9) | (tr1.y.ix[only1].abs() == 9))


class SimpleLinkingTestsIter(CommonTrackingTests):
    def link(self, f, search_range, *args, **kwargs):

        def f_iter(f, first_frame, last_frame):
            """ link_iter requires an (optionally enumerated) generator of
            ndarrays """
            for t in np.arange(first_frame, last_frame + 1,
                               dtype=f['frame'].dtype):
                f_filt = f[f['frame'] == t]
                yield t, f_filt[['y', 'x']].values

        res = f.copy()
        res['particle'] = -1
        for t, ids in link_simple_iter(f_iter(f, 0, int(f['frame'].max())),
                                       search_range, *args, **kwargs):
            res.loc[res['frame'] == t, 'particle'] = ids
        return pandas_sort(res, ['particle', 'frame']).reset_index(drop=True)

    def test_output_dtypes(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                       'frame': np.arange(N)})
        # Integer-typed input
        f['frame'] = f['frame'].astype(np.int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.int)

        # Float-typed input: frame column type is propagated in link_iter
        f['frame'] = f['frame'].astype(np.float)
        actual = self.link(f, 5)
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.float)


class SimpleLinkingTestsDfIter(CommonTrackingTests):
    def link(self, f, search_range, *args, **kwargs):

        def df_iter(f, first_frame, last_frame):
            """ link_df_iter requires a generator of dataframes """
            for t in range(first_frame, last_frame + 1):
                yield f[f['frame'] == t]

        res_iter = link_simple_df_iter(df_iter(f, 0, int(f['frame'].max())),
                                       search_range, *args, **kwargs)
        res = pd.concat(res_iter)
        return pandas_sort(res, ['particle', 'frame']).reset_index(drop=True)

    def test_output_dtypes(self):
        N = 5
        f = DataFrame({'x': np.arange(N), 'y': np.ones(N),
                       'frame': np.arange(N)})
        # Integer-typed input
        f['frame'] = f['frame'].astype(np.int)
        actual = self.link(f, 5)

        # Particle and frame columns should be integer typed
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.int)

        # Float-typed input: frame column type is propagated in link_df_iter
        f['frame'] = f['frame'].astype(np.float)
        actual = self.link(f, 5)
        assert np.issubdtype(actual['particle'], np.int)
        assert np.issubdtype(actual['frame'], np.float)


class TestDropLink(CommonTrackingTests):
    def setUp(self):
        self.linker_opts = dict(link_strategy='drop')

    def test_drop_link(self):
        # One 1D stepper. A new particle appears in frame 2.
        # The resulting subnet causes the trajectory to be broken
        # when link_strategy is 'drop' and search_range is large enough.
        N = 2
        f_1particle = DataFrame({'x': np.arange(N), 'y': np.ones(N), 'frame': np.arange(N)})
        f = f_1particle.append(DataFrame(
            {'x': [3], 'y': [1], 'frame': [1]}), ignore_index=True)
        f_expected_without_subnet = f.copy()
        f_expected_without_subnet['particle'] = [0, 0, 1]
        # The linker assigns new particle IDs in arbitrary order. So
        # comparing with expected values is tricky.
        # We just check for the creation of 2 new trajectories.
        without_subnet = self.link(f, 1.5)
        assert_traj_equal(without_subnet, f_expected_without_subnet)
        with_subnet = self.link(f, 5)
        assert set(with_subnet.particle) == set((0, 1, 2))


class TestNumbaLink(SubnetNeededTests):
    def setUp(self):
        self.linker_opts = dict(link_strategy='numba')


class TestNonrecursiveLink(SubnetNeededTests):
    def setUp(self):
        self.linker_opts = dict(link_strategy='nonrecursive')
