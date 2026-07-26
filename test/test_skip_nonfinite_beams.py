"""Unit tests for Phase 3e Task 3's skip_nonfinite_beams masking logic.

`select_finite_beams` in `particle_filter/particle_filter.py` is a pure
function (no ROS/range_libc dependency), so it is directly unit-testable in
a plain shell:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_skip_nonfinite_beams.py -v

Its semantics are meant to mirror
`scripts/2dlidar/score_sensor_model.apply_skip_nonfinite_mask` exactly (same
finite mask, same "drop entirely" behavior) -- these tests pin that
contract independently (this fork can't import the sibling repo's script,
so the equivalence is asserted by construction/inspection, not a shared
import; see docs/reports/... for the cross-repo reasoning).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from particle_filter.particle_filter import select_finite_beams


def test_all_finite_beams_are_kept_unchanged():
    '''No non-finite beams -> obs/ranges pass through unchanged, n_finite == num_rays.'''
    num_particles, num_rays = 5, 4
    obs = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    ranges_2d = np.arange(num_particles * num_rays, dtype=np.float32).reshape(
        num_particles, num_rays)

    obs_finite, ranges_finite, n_finite = select_finite_beams(obs, ranges_2d)

    assert n_finite == num_rays
    np.testing.assert_array_equal(obs_finite, obs)
    np.testing.assert_array_equal(ranges_finite, ranges_2d.reshape(-1))


def test_inf_beams_and_their_predicted_range_column_are_dropped():
    '''A non-finite obs beam drops both itself AND its predicted-range column,
    for every particle -- not reweighted, removed entirely.'''
    num_particles, num_rays = 3, 4
    obs = np.array([1.0, np.inf, 3.0, np.inf], dtype=np.float32)
    ranges_2d = np.array([
        [10.0, 11.0, 12.0, 13.0],
        [20.0, 21.0, 22.0, 23.0],
        [30.0, 31.0, 32.0, 33.0],
    ], dtype=np.float32)

    obs_finite, ranges_finite, n_finite = select_finite_beams(obs, ranges_2d)

    assert n_finite == 2
    np.testing.assert_array_equal(obs_finite, np.array([1.0, 3.0], dtype=np.float32))
    # particle-major flatten of the surviving columns (0 and 2)
    expected = np.array([10.0, 12.0, 20.0, 22.0, 30.0, 32.0], dtype=np.float32)
    np.testing.assert_array_equal(ranges_finite, expected)


def test_nan_beams_are_also_dropped():
    '''NaN (not just inf) counts as non-finite and is dropped too.'''
    obs = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    ranges_2d = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)

    obs_finite, ranges_finite, n_finite = select_finite_beams(obs, ranges_2d)

    assert n_finite == 2
    np.testing.assert_array_equal(obs_finite, np.array([1.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(ranges_finite, np.array([1.0, 3.0], dtype=np.float32))


def test_all_nonfinite_beams_yields_zero_finite_count():
    '''Degenerate case: every beam is non-finite -> n_finite == 0, empty arrays
    (caller is responsible for the empty-product/uniform-weight fallback).'''
    obs = np.array([np.inf, np.inf, np.nan], dtype=np.float32)
    ranges_2d = np.zeros((2, 3), dtype=np.float32)

    obs_finite, ranges_finite, n_finite = select_finite_beams(obs, ranges_2d)

    assert n_finite == 0
    assert obs_finite.shape == (0,)
    assert ranges_finite.shape == (0,)


def test_output_dtype_is_float32_matching_eval_sensor_model_contract():
    '''eval_sensor_model's pybind signature requires float32 obs/ranges buffers
    (see range_libc pywrapper) -- select_finite_beams must always produce
    float32, even if given float64 input.'''
    obs = np.array([1.0, np.inf, 3.0], dtype=np.float64)
    ranges_2d = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)

    obs_finite, ranges_finite, n_finite = select_finite_beams(obs, ranges_2d)

    assert obs_finite.dtype == np.float32
    assert ranges_finite.dtype == np.float32
    assert n_finite == 2


def test_ranges_finite_is_particle_major_flat_matching_eval_sensor_model_layout():
    '''ranges_finite must flatten as ranges[i*n_finite+j] (particle-major) --
    the same layout eval_sensor_model expects (see RangeLib.h
    eval_sensor_model: ranges[i*rays_per_particle+j]).'''
    num_particles, num_rays = 4, 2
    obs = np.array([1.0, 2.0], dtype=np.float32)  # both finite, no dropping
    ranges_2d = np.arange(num_particles * num_rays, dtype=np.float32).reshape(
        num_particles, num_rays)

    _, ranges_finite, n_finite = select_finite_beams(obs, ranges_2d)

    assert n_finite == num_rays
    for i in range(num_particles):
        for j in range(num_rays):
            assert math.isclose(
                ranges_finite[i * n_finite + j], ranges_2d[i, j], rel_tol=1e-9)
