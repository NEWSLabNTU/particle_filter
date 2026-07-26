"""Unit tests for the Phase 3d diagnostics module.

`particle_filter/diagnostics.py` has no ROS/node imports, so it is
directly unit-testable in a plain shell:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_diagnostics.py -v
"""
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from particle_filter.diagnostics import (
    DiagnosticsRecorder,
    beam_categories,
    effective_sample_size,
    pose_covariance,
    weight_entropy,
)


# --- effective_sample_size (moved here from particle_filter.py) ---

def test_effective_sample_size_uniform_weights_equals_n():
    n = 4000
    weights = np.ones(n) / n
    ess = effective_sample_size(weights)
    assert math.isclose(ess, n, rel_tol=1e-9)


def test_effective_sample_size_one_hot_weights_equals_one():
    n = 4000
    weights = np.zeros(n)
    weights[0] = 1.0
    ess = effective_sample_size(weights)
    assert math.isclose(ess, 1.0, rel_tol=1e-9)


# --- weight_entropy ---

def test_weight_entropy_uniform_weights_equals_ln_n():
    '''Entropy of a uniform distribution over N outcomes is ln(N) (nats).'''
    n = 4000
    weights = np.ones(n) / n
    entropy = weight_entropy(weights)
    assert math.isclose(entropy, math.log(n), rel_tol=1e-9)


def test_weight_entropy_one_hot_weights_equals_zero():
    '''All mass on one particle -> zero entropy (no uncertainty).'''
    n = 4000
    weights = np.zeros(n)
    weights[0] = 1.0
    entropy = weight_entropy(weights)
    assert math.isclose(entropy, 0.0, abs_tol=1e-9)


def test_weight_entropy_normalizes_unnormalized_weights():
    n = 100
    weights = np.full(n, 5.0)
    entropy = weight_entropy(weights)
    assert math.isclose(entropy, math.log(n), rel_tol=1e-9)


# --- pose_covariance ---

def test_pose_covariance_known_weighted_cloud():
    '''Four particles at the corners of a unit square, uniform weights.

    xx and yy variance of {-1,+1} each with mass 0.5 is 1.0 (population
    covariance, matching np.cov(..., aweights=...) semantics used
    elsewhere in this file for /pf/pose/odom). xy covariance is 0 by
    symmetry.
    '''
    particles = np.array([
        [-1.0, -1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    weights = np.ones(4) / 4.0
    cov = pose_covariance(particles, weights)
    assert cov.shape == (2, 2)
    assert math.isclose(cov[0, 0], 1.0, rel_tol=1e-9)
    assert math.isclose(cov[1, 1], 1.0, rel_tol=1e-9)
    assert math.isclose(cov[0, 1], 0.0, abs_tol=1e-9)
    assert math.isclose(cov[1, 0], 0.0, abs_tol=1e-9)


def test_pose_covariance_zero_spread_is_zero():
    particles = np.array([[5.0, 5.0, 0.0]] * 10)
    weights = np.ones(10) / 10.0
    cov = pose_covariance(particles, weights)
    assert math.isclose(cov[0, 0], 0.0, abs_tol=1e-9)
    assert math.isclose(cov[1, 1], 0.0, abs_tol=1e-9)


# --- beam_categories ---

def test_beam_categories_all_five_buckets():
    '''Hand-built case covering hit / short / long / clamped / nonfinite.

    resolution=0.05 m/px, max_range_px=100 (5 m), sigma_px=2 (0.1 m).
    '''
    resolution = 0.05
    max_range_px = 100
    sigma_px = 2.0

    # predicted (d) ranges in metres, all 2.0 m (40 px) except the
    # clamped-check beam which predicts near max range.
    observed_m = np.array([
        2.0,    # hit: matches predicted exactly
        1.5,    # short: well below predicted, outside 3-sigma
        2.5,    # long: well above predicted, outside 3-sigma
        6.0,    # clamped: r/res = 120 px >= max_range_px (100)
        float('inf'),  # nonfinite
    ])
    predicted_m = np.array([2.0, 2.0, 2.0, 2.0, 2.0])

    result = beam_categories(observed_m, predicted_m, resolution,
                              max_range_px, sigma_px)

    assert result['hit'] == pytest_approx(1.0 / 5.0)
    assert result['short'] == pytest_approx(1.0 / 5.0)
    assert result['long'] == pytest_approx(1.0 / 5.0)
    assert result['clamped'] == pytest_approx(1.0 / 5.0)
    assert result['nonfinite'] == pytest_approx(1.0 / 5.0)
    total = sum(result.values())
    assert math.isclose(total, 1.0, rel_tol=1e-9)


def pytest_approx(x, tol=1e-9):
    class _Approx:
        def __eq__(self, other):
            return math.isclose(other, x, abs_tol=tol)
    return _Approx()


def test_beam_categories_all_hits_when_predicted_equals_observed():
    resolution = 0.05
    max_range_px = 1000
    sigma_px = 2.0
    observed_m = np.array([1.0, 2.0, 3.0])
    predicted_m = np.array([1.0, 2.0, 3.0])
    result = beam_categories(observed_m, predicted_m, resolution,
                              max_range_px, sigma_px)
    assert math.isclose(result['hit'], 1.0, rel_tol=1e-9)
    assert math.isclose(result['short'], 0.0, abs_tol=1e-9)
    assert math.isclose(result['long'], 0.0, abs_tol=1e-9)
    assert math.isclose(result['clamped'], 0.0, abs_tol=1e-9)
    assert math.isclose(result['nonfinite'], 0.0, abs_tol=1e-9)


# --- DiagnosticsRecorder ---

def test_diagnostics_recorder_writes_one_json_object_per_line(tmp_path):
    path = str(tmp_path / 'diag.jsonl')
    recorder = DiagnosticsRecorder(path, flush_every=20)
    recorder.record({'iter': 1, 'n_eff': 4000.0})
    recorder.record({'iter': 2, 'n_eff': 3990.5})
    recorder.close()

    with open(path) as f:
        lines = [line for line in f.read().splitlines() if line]

    assert len(lines) == 2
    obj0 = json.loads(lines[0])
    obj1 = json.loads(lines[1])
    assert obj0 == {'iter': 1, 'n_eff': 4000.0}
    assert obj1 == {'iter': 2, 'n_eff': 3990.5}


def test_diagnostics_recorder_flushes_before_flush_every_reached_on_close(tmp_path):
    '''close() must flush even if fewer than flush_every records were made.'''
    path = str(tmp_path / 'diag_small.jsonl')
    recorder = DiagnosticsRecorder(path, flush_every=20)
    recorder.record({'iter': 1})
    recorder.close()

    with open(path) as f:
        lines = [line for line in f.read().splitlines() if line]
    assert len(lines) == 1


def test_diagnostics_recorder_creates_parent_directory(tmp_path):
    path = str(tmp_path / 'nested' / 'dir' / 'diag.jsonl')
    recorder = DiagnosticsRecorder(path, flush_every=20)
    recorder.record({'iter': 1})
    recorder.close()
    assert os.path.exists(path)
