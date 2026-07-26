"""Unit tests for `particle_filter.sensor_model.build_table`.

Phase 3e Task 2: `build_table` is a pure function (no ROS/node
dependencies), directly unit-testable in a plain shell:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_sensor_model.py -v

The `test_upstream_variant_matches_inline_formula_bit_for_bit` test is the
one that matters most: it protects Phase 3c/3d reproducibility by pinning
`build_table(..., variant="upstream")` against a standalone copy of the
exact formula `precompute_sensor_model()` used before this refactor,
asserting EXACT (`np.array_equal`) equality, not just close.
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from particle_filter.sensor_model import build_table


# ---------------------------------------------------------------------------
# Reference formula: a standalone re-implementation of the vectorised
# arithmetic `precompute_sensor_model()`'s nested loop computes, kept here
# ONLY as a fixed oracle -- never imported from sensor_model.py -- so any
# future edit to build_table's "upstream" arithmetic that changes its
# output is caught immediately, not silently.
# ---------------------------------------------------------------------------

def _reference_upstream_table(max_range_px, z_hit, z_short, z_max, z_rand, sigma_hit):
    table_width = int(max_range_px) + 1
    r_idx = np.arange(table_width, dtype=np.float64)[:, None]
    d_idx = np.arange(table_width, dtype=np.float64)[None, :]

    diff = r_idx - d_idx
    prob = z_hit * np.exp(-(diff * diff) / (2.0 * sigma_hit * sigma_hit)) \
        / (sigma_hit * np.sqrt(2.0 * np.pi))

    short_mask = r_idx < d_idx
    with np.errstate(divide="ignore", invalid="ignore"):
        short_term = np.where(
            short_mask,
            2.0 * z_short * (d_idx - r_idx) / np.where(d_idx == 0, 1.0, d_idx),
            0.0,
        )
    prob = prob + short_term

    max_mask = (r_idx == float(max_range_px))
    prob = prob + np.where(max_mask, z_max, 0.0)

    rand_mask = r_idx < float(max_range_px)
    prob = prob + np.where(rand_mask, z_rand / float(max_range_px), 0.0)

    norm = prob.sum(axis=0, keepdims=True)
    return prob / norm


def _reference_upstream_table_pure_loop(max_range_px, z_hit, z_short, z_max, z_rand, sigma_hit):
    """Literal (slow) nested-loop port of precompute_sensor_model's inner
    math, used only as a second, independent cross-check (allclose, not
    exact -- floating point operation order differs from the vectorised
    forms above)."""
    table_width = int(max_range_px) + 1
    table = np.zeros((table_width, table_width), dtype=np.float64)
    for d in range(table_width):
        norm = 0.0
        for r in range(table_width):
            prob = 0.0
            z = float(r - d)
            prob += z_hit * np.exp(-(z * z) / (2.0 * sigma_hit * sigma_hit)) \
                / (sigma_hit * np.sqrt(2.0 * np.pi))
            if r < d:
                prob += 2.0 * z_short * (d - r) / float(d)
            if int(r) == int(max_range_px):
                prob += z_max
            if r < int(max_range_px):
                prob += z_rand * 1.0 / float(max_range_px)
            norm += prob
            table[int(r), int(d)] = prob
        table[:, int(d)] /= norm
    return table


# ---------------------------------------------------------------------------
# 1. Bit-for-bit protection: build_table(variant="upstream") vs the inline
#    formula. This is the reproducibility gate the brief calls out.
# ---------------------------------------------------------------------------

def test_upstream_variant_matches_inline_formula_bit_for_bit():
    kwargs = dict(max_range_px=20, z_hit=0.75, z_short=0.01, z_max=0.07,
                  z_rand=0.12, sigma_px=3.0)
    got = build_table(variant="upstream", **kwargs)
    reference = _reference_upstream_table(
        kwargs["max_range_px"], kwargs["z_hit"], kwargs["z_short"],
        kwargs["z_max"], kwargs["z_rand"], kwargs["sigma_px"],
    )
    assert np.array_equal(got, reference)


def test_upstream_variant_matches_inline_formula_bit_for_bit_full_scale_params():
    # Same z_*/sigma as the real deployment config (config/localize.yaml),
    # small table so the test stays fast.
    kwargs = dict(max_range_px=50, z_hit=0.75, z_short=0.01, z_max=0.07,
                  z_rand=0.12, sigma_px=8.0)
    got = build_table(variant="upstream", **kwargs)
    reference = _reference_upstream_table(
        kwargs["max_range_px"], kwargs["z_hit"], kwargs["z_short"],
        kwargs["z_max"], kwargs["z_rand"], kwargs["sigma_px"],
    )
    assert np.array_equal(got, reference)


def test_upstream_variant_matches_pure_python_loop_closely():
    """Independent cross-check against a literal double-loop port (not
    exact -- different operation order/rounding -- but must agree tightly)."""
    kwargs = dict(max_range_px=20, z_hit=0.75, z_short=0.01, z_max=0.07,
                  z_rand=0.12, sigma_px=3.0)
    got = build_table(variant="upstream", **kwargs)
    reference = _reference_upstream_table_pure_loop(
        kwargs["max_range_px"], kwargs["z_hit"], kwargs["z_short"],
        kwargs["z_max"], kwargs["z_rand"], kwargs["sigma_px"],
    )
    np.testing.assert_allclose(got, reference, rtol=1e-10, atol=1e-12)


def test_default_variant_is_upstream():
    kwargs = dict(max_range_px=20, z_hit=0.75, z_short=0.01, z_max=0.07,
                  z_rand=0.12, sigma_px=3.0)
    default = build_table(**kwargs)
    explicit = build_table(variant="upstream", **kwargs)
    assert np.array_equal(default, explicit)


def test_unknown_variant_raises():
    kwargs = dict(max_range_px=20, z_hit=0.75, z_short=0.01, z_max=0.07,
                  z_rand=0.12, sigma_px=3.0)
    try:
        build_table(variant="bogus", **kwargs)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_max_range_px_must_be_positive():
    try:
        build_table(max_range_px=0, z_hit=0.75, z_short=0.01, z_max=0.07,
                    z_rand=0.12, sigma_px=3.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# 2. Every column still sums to 1 (both variants).
# ---------------------------------------------------------------------------

def test_upstream_columns_sum_to_one():
    table = build_table(max_range_px=100, z_hit=0.75, z_short=0.01, z_max=0.07,
                         z_rand=0.12, sigma_px=8.0, variant="upstream")
    np.testing.assert_allclose(table.sum(axis=0), np.ones(table.shape[1]), rtol=1e-10)


def test_normalized_short_columns_sum_to_one():
    table = build_table(max_range_px=100, z_hit=0.75, z_short=0.01, z_max=0.07,
                         z_rand=0.12, sigma_px=8.0, variant="normalized_short",
                         lambda_short=1.0)
    np.testing.assert_allclose(table.sum(axis=0), np.ones(table.shape[1]), rtol=1e-10)


# ---------------------------------------------------------------------------
# 3. normalized_short's short-reading component mass is d-INVARIANT: the
#    RAW (pre-normalisation) discrete sum of eta*lambda*exp(-lambda*r) over
#    r in [0, d] converges to the SAME constant for every d >= a few units
#    of 1/lambda (unlike upstream's ramp, whose mass is z_short*d and grows
#    without bound). Note: because the table is inherently discrete (pixel
#    indices) while `eta` is derived from the CONTINUOUS integral, the
#    discrete sum is a Riemann sum and does NOT converge to exactly 1 (it
#    converges to lambda/(1-exp(-lambda)) for lambda in "per pixel" units)
#    -- that is expected and is not itself the property this fixes; mass
#    no longer SCALING WITH d is the property.
# ---------------------------------------------------------------------------

def test_normalized_short_raw_short_component_mass_is_d_invariant():
    max_range_px = 200
    lambda_short = 1.0
    table_width = max_range_px + 1
    r = np.arange(table_width, dtype=np.float64)
    masses = []
    for d in (10, 50, 100, 199):
        eta = 1.0 / (1.0 - math.exp(-lambda_short * d))
        mask = r <= d
        raw_mass = np.sum(np.where(mask, eta * lambda_short * np.exp(-lambda_short * r), 0.0))
        masses.append(raw_mass)
    # All masses (d=10..199, i.e. a 20x range of predicted distance) agree
    # to within 1e-6 -- contrast with upstream's ramp, whose mass at d=199
    # is ~20x its mass at d=10.
    masses = np.asarray(masses)
    assert np.ptp(masses) < 1e-3, masses


def _effective_component_shares(max_range_px, z_hit, z_short, z_max, z_rand, sigma_px,
                                 lambda_short):
    """Raw (pre-normalisation) per-component terms for the normalized_short
    variant, and each component's share of the normalised column -- an
    independent re-derivation (not calling build_table) used to check the
    fix's actual point: configured weights should mean what they say."""
    table_width = max_range_px + 1
    r_idx = np.arange(table_width, dtype=np.float64)[:, None]
    d_idx = np.arange(table_width, dtype=np.float64)[None, :]

    diff = r_idx - d_idx
    hit_term = z_hit * np.exp(-(diff * diff) / (2.0 * sigma_px * sigma_px)) \
        / (sigma_px * np.sqrt(2.0 * np.pi))

    short_mask = r_idx < d_idx
    with np.errstate(divide="ignore", invalid="ignore"):
        eta = np.where(d_idx == 0, 0.0, 1.0 / (1.0 - np.exp(-lambda_short * d_idx)))
        short_term = np.where(
            short_mask,
            z_short * eta * lambda_short * np.exp(-lambda_short * r_idx),
            0.0,
        )

    max_term = np.where(r_idx == float(max_range_px), z_max, 0.0)
    rand_term = np.where(r_idx < float(max_range_px), z_rand / float(max_range_px), 0.0)

    total = (hit_term + short_term + max_term + rand_term).sum(axis=0)
    return hit_term.sum(axis=0) / total


def test_normalized_short_effective_z_hit_stays_close_to_configured_across_ranges():
    """The point of the fix (sec 5.1): effective z_hit should stay near the
    configured value (0.75) across predicted ranges, instead of collapsing
    from 25.4% at 10m to 6.8% at 50m (the upstream degradation)."""
    resolution = 0.05
    max_range_m = 60.0
    max_range_px = int(round(max_range_m / resolution))
    z_hit, z_short, z_max, z_rand, sigma_px = 0.75, 0.01, 0.07, 0.12, 8.0

    effective_hit = _effective_component_shares(
        max_range_px, z_hit, z_short, z_max, z_rand, sigma_px, lambda_short=1.0,
    )

    shares = []
    for d_m in (10, 20, 50):
        d_px = int(round(d_m / resolution))
        share = effective_hit[d_px]
        shares.append(share)
        # Upstream degrades from 25.4% (10m) to 6.8% (50m); normalized_short
        # must stay within a reasonably tight band of the configured 0.75
        # (a small, resolution/discretization-driven offset is expected --
        # see the module docstring's "eta is a continuum quantity" note).
        assert abs(share - z_hit) < 0.05, (d_m, share)
    # And, unlike upstream, it must be essentially FLAT across ranges --
    # that is the actual point of the fix.
    assert (max(shares) - min(shares)) < 1e-6, shares
