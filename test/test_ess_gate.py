"""Unit tests for the ESS (effective-sample-size) resampling gate.

Phase 3c Lever 3: `effective_sample_size` and `should_resample` in
`particle_filter/particle_filter.py` are pure functions (no ROS/node
dependencies), so they are directly unit-testable in a plain shell:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_ess_gate.py -v
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from particle_filter.particle_filter import effective_sample_size, should_resample


def test_effective_sample_size_uniform_weights_equals_n():
    '''Uniform weights -> no degeneracy -> N_eff == N (the particle count).'''
    n = 4000
    weights = np.ones(n) / n
    ess = effective_sample_size(weights)
    assert math.isclose(ess, n, rel_tol=1e-9)


def test_effective_sample_size_one_hot_weights_equals_one():
    '''All weight on a single particle -> total degeneracy -> N_eff == 1.'''
    n = 4000
    weights = np.zeros(n)
    weights[0] = 1.0
    ess = effective_sample_size(weights)
    assert math.isclose(ess, 1.0, rel_tol=1e-9)


def test_effective_sample_size_unnormalized_weights_same_as_normalized():
    '''effective_sample_size should normalize internally regardless of input scale.'''
    n = 100
    weights = np.full(n, 5.0)  # unnormalized, but still uniform in shape
    ess = effective_sample_size(weights)
    assert math.isclose(ess, n, rel_tol=1e-9)


def test_should_resample_uniform_weights_no_resample_at_ratio_half():
    '''Uniform weights (N_eff == N) is always >= ratio*N for any ratio <= 1, so no resample.'''
    n = 4000
    weights = np.ones(n) / n
    assert bool(should_resample(weights, n, 0.5)) is False


def test_should_resample_one_hot_weights_resamples():
    '''Fully degenerate weights (N_eff == 1) is far below ratio*N -> resample.'''
    n = 4000
    weights = np.zeros(n)
    weights[0] = 1.0
    assert bool(should_resample(weights, n, 0.5)) is True


def test_should_resample_boundary_case_exactly_at_threshold_does_not_resample():
    '''N_eff exactly equal to ratio*max_particles is NOT below threshold (strict <) -> no resample.'''
    n = 8
    ratio = 0.5
    target_ess = ratio * n  # == 4

    # Construct weights with a known, exact N_eff: k particles share weight
    # 1/k each (rest zero) gives N_eff == k exactly.
    k = int(target_ess)
    weights = np.zeros(n)
    weights[:k] = 1.0 / k

    ess = effective_sample_size(weights)
    assert math.isclose(ess, target_ess, rel_tol=1e-9)
    assert bool(should_resample(weights, n, ratio)) is False


def test_should_resample_just_below_threshold_resamples():
    '''N_eff just below ratio*max_particles -> resample (strict <).'''
    n = 8
    ratio = 0.5
    # k=3 particles sharing weight -> N_eff == 3, below target_ess == 4.
    k = 3
    weights = np.zeros(n)
    weights[:k] = 1.0 / k
    assert bool(should_resample(weights, n, ratio)) is True
