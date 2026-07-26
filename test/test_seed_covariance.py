"""Unit tests for Phase 4 Task 2's covariance-aware /initialpose seeding.

`build_pose_covariance_marginal` and `sample_pose_particles` in
`particle_filter/particle_filter.py` are pure functions (no ROS/node
dependencies), so they are directly unit-testable in a plain shell:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_seed_covariance.py -v

See docs/research/localization/mcl_initialization_and_covariance.md
recommendation 2 for why the covariance marginal must be sampled instead
of discarded.
"""
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from particle_filter.particle_filter import (
    build_pose_covariance_marginal,
    sample_pose_particles,
)


def _zero_cov():
    return [0.0] * 36


def _isotropic_cov(var_xy, var_yaw):
    cov = [0.0] * 36
    cov[0] = var_xy    # xx
    cov[7] = var_xy    # yy
    cov[35] = var_yaw  # yaw-yaw
    return cov


# --- build_pose_covariance_marginal --------------------------------------

def test_extracts_xy_yaw_marginal_at_correct_indices():
    cov = [0.0] * 36
    cov[0] = 1.0   # xx
    cov[1] = 0.2   # xy
    cov[5] = 0.3   # x-yaw
    cov[6] = 0.2   # yx (symmetric)
    cov[7] = 2.0   # yy
    cov[11] = 0.4  # y-yaw
    cov[30] = 0.3  # yaw-x
    cov[31] = 0.4  # yaw-y
    cov[35] = 0.5  # yaw-yaw
    marg = build_pose_covariance_marginal(cov)
    expected = np.array([
        [1.0, 0.2, 0.3],
        [0.2, 2.0, 0.4],
        [0.3, 0.4, 0.5],
    ])
    np.testing.assert_allclose(marg, expected)


def test_asymmetric_input_is_symmetrised_without_warning():
    '''Some publishers round asymmetrically -- e.g. cov[1] != cov[6].
    build_pose_covariance_marginal must average the pair and produce a
    matrix numpy's multivariate_normal accepts without a RuntimeWarning.'''
    cov = [0.0] * 36
    cov[0] = 1.0
    cov[7] = 1.0
    cov[35] = 0.5
    cov[1] = 0.10   # xy
    cov[6] = 0.14   # yx, deliberately different from cov[1]
    marg = build_pose_covariance_marginal(cov)
    np.testing.assert_allclose([marg[0, 1], marg[1, 0]], [0.12, 0.12])
    np.testing.assert_allclose(marg, marg.T)

    with warnings.catch_warnings():
        warnings.simplefilter('error')
        # must not raise/warn now that the input is symmetric
        np.random.default_rng(0).multivariate_normal(
            np.zeros(3), marg, size=100, check_valid='raise')


def test_ignores_z_roll_pitch_terms():
    '''z/roll/pitch rows/cols must not leak into the (x,y,yaw) marginal.'''
    cov = [0.0] * 36
    cov[0] = 1.0
    cov[7] = 1.0
    cov[35] = 0.5
    cov[14] = 999.0  # z-z, should be ignored
    cov[21] = 999.0  # roll-roll, should be ignored
    cov[28] = 999.0  # pitch-pitch, should be ignored
    marg = build_pose_covariance_marginal(cov)
    np.testing.assert_allclose(marg, [[1.0, 0, 0], [0, 1.0, 0], [0, 0, 0.5]])


# --- sample_pose_particles: fallback path --------------------------------

def test_zero_covariance_takes_fallback_path():
    particles, path = sample_pose_particles(
        1.0, 2.0, 0.3, _zero_cov(), 500, 0.5, 0.4,
        rng=np.random.default_rng(1))
    assert path == 'fallback'
    assert particles.shape == (500, 3)


def test_none_covariance_takes_fallback_path():
    particles, path = sample_pose_particles(
        1.0, 2.0, 0.3, None, 500, 0.5, 0.4,
        rng=np.random.default_rng(1))
    assert path == 'fallback'


def test_fallback_path_matches_todays_independent_scalar_sampling():
    '''Behaviour must be identical to today (independent per-axis Gaussian,
    x/y/theta order) when covariance is absent/zero -- this is what keeps
    the INITPOSE_SOURCE=gt_bag oracle path byte-for-byte reproducible.'''
    n = 2000
    x0, y0, yaw0 = 5.0, -3.0, 1.2
    xy_sigma, theta_sigma = 0.5, 0.4

    rng1 = np.random.default_rng(7)
    particles, path = sample_pose_particles(
        x0, y0, yaw0, _zero_cov(), n, xy_sigma, theta_sigma, rng=rng1)
    assert path == 'fallback'

    rng2 = np.random.default_rng(7)
    expected = np.empty((n, 3))
    expected[:, 0] = x0 + rng2.normal(loc=0.0, scale=xy_sigma, size=n)
    expected[:, 1] = y0 + rng2.normal(loc=0.0, scale=xy_sigma, size=n)
    expected[:, 2] = yaw0 + rng2.normal(loc=0.0, scale=theta_sigma, size=n)

    np.testing.assert_allclose(particles, expected)


def test_non_positive_semidefinite_matrix_falls_back():
    '''A matrix that isn't a valid covariance (not PSD) must not crash --
    fall back to the scalar sampling instead.'''
    cov = [0.0] * 36
    cov[0] = 1.0
    cov[7] = 1.0
    cov[35] = 1.0
    cov[1] = cov[6] = 100.0  # xy correlation far exceeds sqrt(xx*yy) -> not PSD
    particles, path = sample_pose_particles(
        0.0, 0.0, 0.0, cov, 200, 0.5, 0.4, rng=np.random.default_rng(3))
    assert path == 'fallback'
    assert particles.shape == (200, 3)
    assert np.all(np.isfinite(particles))


# --- sample_pose_particles: covariance path ------------------------------

def test_isotropic_covariance_matches_requested_variance():
    '''Known covariance in -> sampled cloud whose empirical covariance
    matches to a reasonable tolerance with a fixed seed.'''
    n = 200000
    var_xy, var_yaw = 4.0, 1.0  # matches the measured GNSS covariance
    x0, y0, yaw0 = 10.0, 20.0, 0.5
    particles, path = sample_pose_particles(
        x0, y0, yaw0, _isotropic_cov(var_xy, var_yaw), n, 0.5, 0.4,
        rng=np.random.default_rng(42))
    assert path == 'covariance'

    empirical_mean = particles.mean(axis=0)
    np.testing.assert_allclose(
        empirical_mean, [x0, y0, yaw0], atol=0.05)

    empirical_cov = np.cov(particles, rowvar=False)
    expected_cov = np.diag([var_xy, var_xy, var_yaw])
    np.testing.assert_allclose(empirical_cov, expected_cov, atol=0.05)


def test_correlated_covariance_produces_nonzero_empirical_correlation():
    '''Correlated covariance -> non-zero empirical correlation (proves the
    off-diagonal terms actually reach the sampler, unlike the old
    independent-per-axis draw which could never produce this).'''
    n = 200000
    cov = [0.0] * 36
    cov[0] = 4.0    # xx
    cov[1] = 3.0    # xy (strong positive correlation)
    cov[6] = 3.0
    cov[7] = 4.0    # yy
    cov[35] = 1.0   # yaw-yaw
    particles, path = sample_pose_particles(
        0.0, 0.0, 0.0, cov, n, 0.5, 0.4, rng=np.random.default_rng(5))
    assert path == 'covariance'

    empirical_cov = np.cov(particles, rowvar=False)
    # requested xy covariance is 3.0; must be clearly nonzero and of the
    # correct sign, not collapsed to ~0 as independent sampling would give.
    assert empirical_cov[0, 1] > 2.5
    correlation = empirical_cov[0, 1] / np.sqrt(
        empirical_cov[0, 0] * empirical_cov[1, 1])
    assert correlation > 0.6  # requested correlation is 3/sqrt(4*4) = 0.75


def test_covariance_particle_count_and_yaw_offset_applied():
    n = 1000
    particles, path = sample_pose_particles(
        0.0, 0.0, 2.5, _isotropic_cov(1.0, 0.5), n, 0.5, 0.4,
        rng=np.random.default_rng(9))
    assert path == 'covariance'
    assert particles.shape == (n, 3)
    # yaw column must be centred on the seed yaw, not on 0
    np.testing.assert_allclose(particles[:, 2].mean(), 2.5, atol=0.1)
