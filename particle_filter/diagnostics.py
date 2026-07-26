'''Phase 3d Task 1: per-update MCL diagnostics.

Pure helpers (no ROS/rclpy imports) plus a small JSONL writer, so this
module is directly unit-testable in a plain shell:

    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest test/test_diagnostics.py -v

`effective_sample_size` used to live in `particle_filter.py` (added in
Phase 3c Lever 3 for the ESS resampling gate). It now lives here as the
single definition; `particle_filter.py` imports and re-exports it so
existing callers (including `should_resample` and the Lever-3 tests)
keep working unchanged.
'''
import json
import os

import numpy as np


def effective_sample_size(weights):
    '''
    Compute the effective sample size (N_eff) of a set of (not necessarily
    normalized) importance weights: N_eff = 1 / sum(w_normalized^2).

    N_eff == N (the particle count) when weights are uniform (best case,
    no degeneracy) and N_eff == 1 when a single particle carries all the
    weight (worst case, total degeneracy). Pure function -- no ROS/node
    dependencies -- so it is directly unit-testable.
    '''
    weights = np.asarray(weights, dtype=np.float64)
    total = np.sum(weights)
    if total <= 0.0:
        return 0.0
    normalized = weights / total
    sum_sq = np.sum(normalized * normalized)
    if sum_sq <= 0.0:
        return 0.0
    return 1.0 / sum_sq


def weight_entropy(weights):
    '''
    Compute the Shannon entropy (in nats) of a set of (not necessarily
    normalized) importance weights: H = -sum(p * ln(p)) over normalized
    weights p, with the convention 0*ln(0) = 0.

    H == ln(N) for uniform weights (maximum uncertainty / diversity) and
    H == 0 when a single particle carries all the weight (no
    uncertainty). Pure function, directly unit-testable.
    '''
    weights = np.asarray(weights, dtype=np.float64)
    total = np.sum(weights)
    if total <= 0.0:
        return 0.0
    normalized = weights / total
    nonzero = normalized[normalized > 0.0]
    return float(-np.sum(nonzero * np.log(nonzero)))


def pose_covariance(particles, weights):
    '''
    Compute the 2x2 weighted covariance of the particle cloud's (x, y)
    positions.

    `particles` is an (N, 3+) array whose first two columns are x, y;
    `weights` is an (N,) array of (not necessarily normalized) importance
    weights. Uses population (ddof=0) weighted covariance, matching the
    `np.cov(..., aweights=...)` call already used for the published
    `/pf/pose/odom` covariance in `particle_filter.py`. Pure function,
    directly unit-testable.
    '''
    particles = np.asarray(particles, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    xy = particles[:, 0:2]
    cov = np.cov(xy, rowvar=False, ddof=0, aweights=weights)
    return np.asarray(cov).reshape(2, 2)


def beam_categories(observed_m, predicted_m, resolution, max_range_px,
                     sigma_px):
    '''
    Categorize each beam of a scan into one of five mutually-exclusive,
    exhaustive buckets, returning a dict of fractions that sum to 1.0
    (or all-zero if there are no beams):

    - `nonfinite`: the observed range is not finite (e.g. `inf`, no
      return).
    - `clamped`: the observed range, in pixels, is at or beyond
      `max_range_px`.
    - `hit`: the observed and predicted ranges (in pixels) agree within
      `3 * sigma_px`.
    - `short`: observed range is below predicted, outside the hit band
      (undershoot -- e.g. an obstacle nearer than the map expects).
    - `long`: observed range is above predicted, outside the hit band
      (overshoot).

    `observed_m`/`predicted_m` are in metres; `resolution` (m/px),
    `max_range_px`, and `sigma_px` put the hit/clamp thresholds in the
    same pixel units used by the sensor model table. Pure function,
    directly unit-testable.
    '''
    observed_m = np.asarray(observed_m, dtype=np.float64)
    predicted_m = np.asarray(predicted_m, dtype=np.float64)
    n = observed_m.shape[0]

    keys = ('hit', 'short', 'long', 'clamped', 'nonfinite')
    if n == 0:
        return {k: 0.0 for k in keys}

    r_px = observed_m / resolution
    d_px = predicted_m / resolution

    finite = np.isfinite(r_px)
    nonfinite_mask = ~finite

    # Only classify finite beams beyond this point; nonfinite beams are
    # already accounted for.
    clamped_mask = finite & (r_px >= max_range_px)

    remaining = finite & ~clamped_mask
    diff = r_px - d_px
    hit_mask = remaining & (np.abs(diff) <= 3.0 * sigma_px)
    short_mask = remaining & ~hit_mask & (diff < 0.0)
    long_mask = remaining & ~hit_mask & (diff >= 0.0)

    counts = {
        'hit': int(np.count_nonzero(hit_mask)),
        'short': int(np.count_nonzero(short_mask)),
        'long': int(np.count_nonzero(long_mask)),
        'clamped': int(np.count_nonzero(clamped_mask)),
        'nonfinite': int(np.count_nonzero(nonfinite_mask)),
    }
    return {k: counts[k] / float(n) for k in keys}


class DiagnosticsRecorder:
    '''
    Appends one JSON object per line (JSONL) to `path`, flushing every
    `flush_every` records (and always on `close()`). No ROS imports, so
    it is directly unit-testable in a plain shell.
    '''

    def __init__(self, path, flush_every=20):
        self.path = path
        self.flush_every = flush_every
        self._pending = 0
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._file = open(path, 'a')

    def record(self, record):
        '''Append one record (a JSON-serializable dict) as a line.'''
        self._file.write(json.dumps(record) + '\n')
        self._pending += 1
        if self._pending >= self.flush_every:
            self._file.flush()
            self._pending = 0

    def close(self):
        '''Flush and close the underlying file.'''
        self._file.flush()
        self._file.close()
