"""Unit tests for Phase 3e Task 4's scan-gated correction step.

`compose_odometry_delta` and `should_run_correction` in
`particle_filter/particle_filter.py` are pure functions (no ROS/node
dependencies), so they are directly unit-testable in a plain shell:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_update_on_new_scan_only.py -v

See docs/research/localization/2d_mcl_algorithm.md sec 5.3 for why the
correction needs to run at scan rate (~10 Hz), not odom rate (~20 Hz).
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from particle_filter.particle_filter import (
    compose_odometry_delta,
    select_finite_beams,
    should_run_correction,
)


# --- compose_odometry_delta ---------------------------------------------

def test_compose_from_zero_accumulator_matches_local_delta_exactly():
    '''accum=[0,0,0] (the always-corrected/default path, since odometry_data
    is reset to zero after every single-step correction) must reduce to the
    old overwrite-on-every-odomCB assignment exactly.'''
    local_delta = np.array([0.21, -0.05, 0.013])
    result = compose_odometry_delta(np.zeros(3), local_delta)
    np.testing.assert_allclose(result, local_delta, atol=1e-12)


def test_compose_straight_line_motion_sums_linearly():
    '''No heading change between steps (dtheta=0 each step) -> xy deltas
    just add, since the rotation is identity throughout.'''
    step1 = np.array([1.0, 0.0, 0.0])
    step2 = np.array([1.0, 0.0, 0.0])
    accum = compose_odometry_delta(np.zeros(3), step1)
    accum = compose_odometry_delta(accum, step2)
    np.testing.assert_allclose(accum, [2.0, 0.0, 0.0], atol=1e-12)


def test_compose_two_steps_matches_direct_two_step_odometry_computation():
    '''Exact-composition check: build a 3-pose odometry trajectory (p0,th0),
    (p1,th1), (p2,th2), compute per-step local deltas the same way odomCB
    does (rotate the world-frame delta into the frame of the *preceding*
    pose), fold them with compose_odometry_delta, and confirm the result
    equals the single-shot delta computed directly in the frame of pose 0
    (i.e. R(-th0) @ (p2 - p0), dtheta = th2 - th0). This is the algebraic
    identity the Task 4 report claims (exact, not approximated).'''
    p0, th0 = np.array([0.0, 0.0]), 0.0
    p1, th1 = np.array([1.0, 0.3]), 0.4
    p2, th2 = np.array([2.2, 0.9]), 0.7

    def rot(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s], [s, c]])

    def local_delta(p_from, th_from, p_to, th_to):
        xy = rot(-th_from) @ (p_to - p_from)
        return np.array([xy[0], xy[1], th_to - th_from])

    step1 = local_delta(p0, th0, p1, th1)
    step2 = local_delta(p1, th1, p2, th2)

    accum = compose_odometry_delta(np.zeros(3), step1)
    accum = compose_odometry_delta(accum, step2)

    expected_xy = rot(-th0) @ (p2 - p0)
    expected = np.array([expected_xy[0], expected_xy[1], th2 - th0])

    np.testing.assert_allclose(accum, expected, atol=1e-10)


def test_compose_is_associative_across_many_small_steps():
    '''Folding N small random steps one at a time must match composing the
    single big delta between the trajectory endpoints directly (same
    identity as above, generalized to N>2 steps) -- confirms the running
    accumulator doesn't drift from repeated rotation composition.'''
    rng = np.random.default_rng(42)
    n_steps = 20
    poses = [np.array([0.0, 0.0])]
    thetas = [0.0]
    for _ in range(n_steps):
        poses.append(poses[-1] + rng.normal(scale=0.2, size=2))
        thetas.append(thetas[-1] + rng.normal(scale=0.05))

    def rot(theta):
        c, s = math.cos(theta), math.sin(theta)
        return np.array([[c, -s], [s, c]])

    accum = np.zeros(3)
    for i in range(n_steps):
        xy = rot(-thetas[i]) @ (poses[i + 1] - poses[i])
        step = np.array([xy[0], xy[1], thetas[i + 1] - thetas[i]])
        accum = compose_odometry_delta(accum, step)

    expected_xy = rot(-thetas[0]) @ (poses[-1] - poses[0])
    expected = np.array([expected_xy[0], expected_xy[1], thetas[-1] - thetas[0]])

    np.testing.assert_allclose(accum, expected, atol=1e-9)


# --- should_run_correction -----------------------------------------------

def test_flag_off_always_runs_regardless_of_scan_stamps():
    '''Default behavior (flag False): always correct, matching upstream's
    "correct on every odomCB" -- even with no scan seen yet.'''
    assert should_run_correction(False, None, None) is True
    assert should_run_correction(False, (5, 0), (5, 0)) is True


def test_flag_on_no_scan_yet_does_not_run():
    assert should_run_correction(True, None, None) is False


def test_flag_on_new_scan_runs_once():
    '''A scan not yet consumed by a correction -> run.'''
    assert should_run_correction(True, (5, 0), None) is True
    assert should_run_correction(True, (5, 0), (4, 0)) is True


def test_flag_on_already_consumed_scan_does_not_rerun():
    '''Same scan stamp as the last correction -> odom-rate calls between
    scans must not re-trigger the correction (this is the fix for sec 5.3's
    double-Bayes-update).'''
    assert should_run_correction(True, (5, 0), (5, 0)) is False


# --- min_finite_beams (reuses select_finite_beams from Task 3) -----------

def test_min_finite_beams_threshold_semantics_via_select_finite_beams():
    '''The particle_filter.py caller compares n_finite < MIN_FINITE_BEAMS;
    pin the boundary semantics against select_finite_beams' n_finite output
    (>= threshold keeps evaluating, < threshold triggers the guard).'''
    min_finite_beams = 10
    obs_ok = np.array([1.0] * 10, dtype=np.float32)
    obs_ok[0] = np.inf  # 9 finite -> below threshold
    ranges_2d = np.ones((3, 10), dtype=np.float32)

    _, _, n_finite = select_finite_beams(obs_ok, ranges_2d)
    assert n_finite == 9
    assert n_finite < min_finite_beams  # guard should trigger

    obs_ok2 = np.array([1.0] * 11, dtype=np.float32)
    obs_ok2[0] = np.inf  # 10 finite -> at threshold, not below
    ranges_2d2 = np.ones((3, 11), dtype=np.float32)
    _, _, n_finite2 = select_finite_beams(obs_ok2, ranges_2d2)
    assert n_finite2 == 10
    assert not (n_finite2 < min_finite_beams)  # guard should NOT trigger
