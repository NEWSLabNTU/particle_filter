"""Sensor-model (beam/measurement model) table construction.

Phase 3e Task 2: moved out of `precompute_sensor_model()`
(`particle_filter.py`) into a pure function so it is unit-testable without a
ROS node/context. See `docs/research/localization/2d_mcl_algorithm.md`
sec 3.1 (how the table is built and consumed) and sec 5.1 (the `p_short`
normalisation defect the `normalized_short` variant below fixes).

Units: `max_range_px` and `sigma_px` are in MAP PIXELS, not metres --
`precompute_sensor_model` passes `self.MAX_RANGE_PX` / `self.SIGMA_HIT`,
both already pixel quantities (map resolution silently retunes the model;
see the algorithm doc sec 3.1). `lambda_short` (used only by the
`normalized_short` variant) is likewise a 1/PIXEL rate, applied directly to
raw pixel offsets `r`/`d` -- its effective decay length in METRES is
`1 / (lambda_short * resolution_m_per_px)`. A `lambda_short` tuned for one
map resolution therefore means something different at another (halving the
cell size halves the decay length in pixels for a fixed `lambda_short`, so
halve `lambda_short` too to hold the metric decay length constant).

variant="upstream": bit-for-bit the same arithmetic as the original
`precompute_sensor_model` nested loop, re-expressed with numpy
broadcasting instead of a double `for` loop (same terms, same conditions,
same per-column normalisation). This is the ONLY variant used in
production before Phase 3e Task 2 lands; `test/test_sensor_model.py`
pins it against a literal copy of that formula to protect Phase 3c/3d
reproducibility.

variant="normalized_short": the short-reading component is replaced with a
per-column-normalised truncated exponential (the canonical textbook fix,
sec 5.1's "Canonical fix"):

    p_short(r|d) = eta * lambda_short * exp(-lambda_short * r)   for 0 <= r <= d
    eta = 1 / (1 - exp(-lambda_short * d))

Unlike the upstream ramp (`2*z_short*(d-r)/d`, which integrates to
`z_short*d` -- mass that grows linearly with the predicted range in
pixels), this component's total mass over `r in [0, d]` is exactly 1
regardless of `d`, so the configured `z_short` weight means what it says
at every predicted range.
"""
import numpy as np

VARIANTS = ("upstream", "normalized_short")


def build_table(max_range_px, z_hit, z_short, z_max, z_rand, sigma_px,
                 variant="upstream", lambda_short=1.0):
    """Build the normalised (r, d) sensor-model table in PIXEL units.

    Returns a float64 array of shape (table_width, table_width),
    table_width = max_range_px + 1, table[r, d] = P(observe r | predicted
    d), each column normalised to sum to 1.

    `variant`: "upstream" (default, matches today's `precompute_sensor_model`
    bit-for-bit) or "normalized_short" (see module docstring).
    `lambda_short`: 1/pixel decay rate for the "normalized_short" variant's
    truncated-exponential short-reading component; ignored for "upstream".
    """
    if variant not in VARIANTS:
        raise ValueError(
            f"build_table: unknown variant {variant!r}; available: {VARIANTS}"
        )
    if max_range_px < 1:
        raise ValueError("build_table: max_range_px must be >= 1")

    table_width = int(max_range_px) + 1
    r_idx = np.arange(table_width, dtype=np.float64)[:, None]   # rows: observed r
    d_idx = np.arange(table_width, dtype=np.float64)[None, :]   # cols: predicted d

    diff = r_idx - d_idx
    hit_term = z_hit * np.exp(-(diff * diff) / (2.0 * sigma_px * sigma_px)) \
        / (sigma_px * np.sqrt(2.0 * np.pi))

    short_mask = r_idx < d_idx
    if variant == "upstream":
        with np.errstate(divide="ignore", invalid="ignore"):
            short_term = np.where(
                short_mask,
                2.0 * z_short * (d_idx - r_idx) / np.where(d_idx == 0, 1.0, d_idx),
                0.0,
            )
    else:  # normalized_short
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = 1.0 - np.exp(-lambda_short * d_idx)
            eta = np.where(d_idx == 0, 0.0, 1.0 / np.where(d_idx == 0, 1.0, denom))
            short_term = np.where(
                short_mask,
                z_short * eta * lambda_short * np.exp(-lambda_short * r_idx),
                0.0,
            )

    prob = hit_term + short_term

    max_mask = (r_idx == float(max_range_px))
    prob = prob + np.where(max_mask, z_max, 0.0)

    rand_mask = r_idx < float(max_range_px)
    prob = prob + np.where(rand_mask, z_rand / float(max_range_px), 0.0)

    norm = prob.sum(axis=0, keepdims=True)
    return prob / norm
