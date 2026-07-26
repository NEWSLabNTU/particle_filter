# MIT License

# Copyright (c) 2020 Hongrui Zheng, Corey Walsh

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the 'Software'), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ros2 python
import rclpy
from rclpy.node import Node

# libraries
import numpy as np
import os
import range_libc
import time
from threading import Lock
from particle_filter import utils as Utils
from particle_filter.sensor_model import build_table
from particle_filter.diagnostics import (
    DiagnosticsRecorder,
    beam_categories,
    effective_sample_size,
    pose_covariance,
    weight_entropy,
)

# TF
# import tf.transformations
# import tf
from tf2_ros import TransformBroadcaster
import tf_transformations

# messages
from std_msgs.msg import String, Header, Float32MultiArray, Float32
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped, Polygon, Point32, PoseWithCovarianceStamped, PointStamped, TransformStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.srv import GetMap

'''
These flags indicate several variants of the sensor model. Only one of them is used at a time.
'''
VAR_NO_EVAL_SENSOR_MODEL = 0
VAR_CALC_RANGE_MANY_EVAL_SENSOR = 1
VAR_REPEAT_ANGLES_EVAL_SENSOR = 2
VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT = 3
VAR_RADIAL_CDDT_OPTIMIZATIONS = 4


# NOTE: effective_sample_size() lives in diagnostics.py (Phase 3d Task 1
# moved it there to share a single definition with the diagnostics
# recorder); it is imported above and re-exported here so existing
# callers (should_resample below, and test/test_ess_gate.py, which
# imports it from this module) keep working unchanged.


def select_finite_beams(obs, ranges_2d):
    '''
    Phase 3e Task 3: pure masking logic `_eval_sensor_model_skip_nonfinite`
    applies before calling `range_method.eval_sensor_model` -- drop
    non-finite observed beams (`obs`) and their paired predicted-range
    columns (`ranges_2d`, shape (num_particles, num_rays)) entirely.
    Mirrors `scripts/2dlidar/score_sensor_model.apply_skip_nonfinite_mask`'s
    masking of (observed_ranges_m, downsampled_angles): same finite mask,
    same "drop entirely, don't reweight" semantics -- just applied to the
    online per-particle predicted-range matrix instead of a single grid's
    predicted-range vector, so the offline gate's score is a truthful
    prediction of this function's effect on the running filter.

    Returns `(obs_finite, ranges_finite_flat, n_finite)`: `obs_finite` is
    a 1D float32 array (n_finite,); `ranges_finite_flat` is a 1D float32
    array (num_particles*n_finite,), particle-major (matching
    `eval_sensor_model`'s expected layout); `n_finite` is the surviving
    beam count (0 if every beam this update is non-finite).

    Pure function -- no range_libc/ROS dependency -- directly
    unit-testable.
    '''
    obs = np.asarray(obs)
    ranges_2d = np.asarray(ranges_2d)
    finite_mask = np.isfinite(obs)
    n_finite = int(np.count_nonzero(finite_mask))
    obs_finite = np.ascontiguousarray(obs[finite_mask], dtype=np.float32)
    ranges_finite = np.ascontiguousarray(
        ranges_2d[:, finite_mask], dtype=np.float32).reshape(-1)
    return obs_finite, ranges_finite, n_finite


def should_resample(weights, max_particles, ratio):
    '''
    Decide whether to resample given the current importance weights: resample
    only when the effective sample size drops below `ratio * max_particles`
    (the ESS gate). Pure function, directly unit-testable.
    '''
    return effective_sample_size(weights) < (ratio * max_particles)


def compose_odometry_delta(accum, local_delta):
    '''
    Phase 3e Task 4: fold one odomCB-computed body-frame delta
    (`local_delta`, [dx, dy, dtheta] expressed in the coordinate frame of
    the pose *just before* this delta -- i.e. the last odometry sample's
    heading) into a running accumulator (`accum`, same layout, expressed
    in the coordinate frame of the pose as of the last MCL correction).

    This is an EXACT composition, not a small-angle approximation:
    `local_delta` is always computed (in odomCB) relative to the
    immediately preceding odometry sample's heading, and `accum[2]` is
    exactly that sample's heading offset from the reference frame (by
    induction -- see docs/reports/... derivation in the Task 4 report).
    Rotating `local_delta`'s xy by `accum[2]` before adding brings it into
    the reference frame precisely; summing raw (unrotated) xy components
    instead would introduce an error of order |local_delta_xy| * accum[2]
    per fold, which this function avoids entirely.

    When `accum` starts at [0, 0, 0] (the default, always-corrected path)
    this reduces to `local_delta` unchanged (cos(0)=1, sin(0)=0) -- i.e.
    it is a strict, behavior-preserving generalization of the old
    overwrite-on-every-odomCB assignment, not a new code path that only
    activates under the new flag.

    Pure function -- no ROS/numpy-Node dependency beyond arrays -- directly
    unit-testable.
    '''
    accum = np.asarray(accum, dtype=np.float64)
    local_delta = np.asarray(local_delta, dtype=np.float64)
    c = np.cos(accum[2])
    s = np.sin(accum[2])
    dx, dy, dtheta = local_delta[0], local_delta[1], local_delta[2]
    return np.array([
        accum[0] + c * dx - s * dy,
        accum[1] + s * dx + c * dy,
        accum[2] + dtheta,
    ])


def should_run_correction(update_on_new_scan_only, last_scan_stamp,
                           last_corrected_scan_stamp):
    '''
    Phase 3e Task 4: decide whether `update()` should run the full MCL
    correction (motion + sensor model + publish) this call.

    - Flag off (default): always True -- reproduces upstream's
      "correct on every odomCB" behavior exactly.
    - Flag on: True only when a scan has arrived (`last_scan_stamp` is set)
      that hasn't already been consumed by a prior correction
      (`last_scan_stamp != last_corrected_scan_stamp`) -- i.e. the
      correction now runs at scan rate instead of odom rate, fixing the
      double-Bayes-update described in
      docs/research/localization/2d_mcl_algorithm.md sec 5.3.

    Pure function, directly unit-testable.
    '''
    if not update_on_new_scan_only:
        return True
    if last_scan_stamp is None:
        return False
    return last_scan_stamp != last_corrected_scan_stamp


class ParticleFiler(Node):
    '''
    This class implements Monte Carlo Localization based on odometry and a laser scanner.
    '''

    def __init__(self):
        super().__init__('particle_filter')

        # declare parameters
        self.declare_parameter('angle_step')
        self.declare_parameter('max_particles')
        self.declare_parameter('max_viz_particles')
        self.declare_parameter('squash_factor')
        self.declare_parameter('max_range')
        self.declare_parameter('theta_discretization')
        self.declare_parameter('range_method')
        self.declare_parameter('rangelib_variant')
        self.declare_parameter('fine_timing')
        self.declare_parameter('publish_odom')
        self.declare_parameter('viz')
        self.declare_parameter('z_short')
        self.declare_parameter('z_max')
        self.declare_parameter('z_rand')
        self.declare_parameter('z_hit')
        self.declare_parameter('sigma_hit')
        self.declare_parameter('sensor_model_variant', 'upstream')
        self.declare_parameter('sensor_model_lambda_short', 1.0)
        self.declare_parameter('skip_nonfinite_beams', False)
        self.declare_parameter('min_finite_beams', 10)
        self.declare_parameter('update_on_new_scan_only', False)
        self.declare_parameter('motion_dispersion_x')
        self.declare_parameter('motion_dispersion_y')
        self.declare_parameter('motion_dispersion_theta')
        self.declare_parameter('scan_topic')
        self.declare_parameter('odometry_topic')
        self.declare_parameter('use_ess_gate', False)
        self.declare_parameter('ess_threshold_ratio', 0.5)
        self.declare_parameter('diag_enable', False)
        self.declare_parameter('diag_path', '')
        self.declare_parameter('diag_every', 1)
        self.declare_parameter('diag_beam_arrays', False)
        self.declare_parameter('diag_topics', False)
        self.declare_parameter('likelihood_field_enable', False)
        self.declare_parameter('lf_window_m', 40.0)
        self.declare_parameter('lf_res_m', 0.5)
        self.declare_parameter('lf_period_s', 1.0)
        self.declare_parameter('lf_log_floor', 20.0)
        self.declare_parameter('random_seed', -1)

        # parameters
        self.ANGLE_STEP           = self.get_parameter('angle_step').value
        self.MAX_PARTICLES        = self.get_parameter('max_particles').value
        self.MAX_VIZ_PARTICLES    = self.get_parameter('max_viz_particles').value
        self.INV_SQUASH_FACTOR    = 1.0 / self.get_parameter('squash_factor').value
        self.MAX_RANGE_METERS     = self.get_parameter('max_range').value
        self.THETA_DISCRETIZATION = self.get_parameter('theta_discretization').value
        self.WHICH_RM             = self.get_parameter('range_method').value
        self.RANGELIB_VAR         = self.get_parameter('rangelib_variant').value
        self.SHOW_FINE_TIMING     = self.get_parameter('fine_timing').value
        self.PUBLISH_ODOM         = self.get_parameter('publish_odom').value
        self.DO_VIZ               = self.get_parameter('viz').value

        # Phase 3e Task 5: optional reproducibility seed for numpy's global
        # RNG (used throughout this module for particle init/resampling/
        # motion noise -- see np.random.* call sites). Default (-1) means
        # "do not seed" -- today's behavior, unchanged: numpy falls back to
        # its own OS-entropy-seeded global state, so consecutive runs are
        # not reproducible. RANDOM_SEED >= 0 calls np.random.seed() once,
        # here, before any particle initialization or subscription
        # callback can consume randomness, so a given seed deterministically
        # reproduces a run (module-global RNG state, so only meaningful
        # with a single particle_filter process per interpreter).
        self.RANDOM_SEED = self.get_parameter('random_seed').value
        if self.RANDOM_SEED is not None and self.RANDOM_SEED >= 0:
            np.random.seed(self.RANDOM_SEED)
            self.get_logger().info('Seeded numpy RNG with random_seed=%d' % self.RANDOM_SEED)

        # effective-sample-size (ESS) resampling gate (Phase 3c Lever 3):
        # when enabled, resample only when N_eff falls below
        # ess_threshold_ratio * max_particles, instead of every update.
        # Default preserves upstream behavior (always resample).
        self.USE_ESS_GATE         = self.get_parameter('use_ess_gate').value
        self.ESS_THRESHOLD_RATIO  = self.get_parameter('ess_threshold_ratio').value

        # per-update diagnostics recorder (Phase 3d Task 1): disabled by
        # default (no-op, no file created, no measurable overhead --
        # matches upstream/Phase 3c behavior exactly).
        self.DIAG_ENABLE       = self.get_parameter('diag_enable').value
        self.DIAG_PATH         = self.get_parameter('diag_path').value
        self.DIAG_EVERY        = self.get_parameter('diag_every').value
        self.DIAG_BEAM_ARRAYS  = self.get_parameter('diag_beam_arrays').value
        self.diag_recorder = None
        if self.DIAG_ENABLE:
            diag_path = self.DIAG_PATH
            if not diag_path:
                diag_path = os.path.join('.', 'tmp', 'mcl_diag_%d.jsonl' % os.getpid())
            self.diag_recorder = DiagnosticsRecorder(diag_path)
            self.get_logger().info('Diagnostics enabled, recording to: ' + diag_path)
        self._diag_last_wall = None

        # live diagnostic scalar topics (Phase 3d Task 2): disabled by
        # default (no publishers created, no measurable overhead). Reuses
        # the same per-update values computed for the JSONL recorder above
        # -- see record_diagnostics() -- so PlotJuggler can subscribe to
        # these directly without a rosbag round-trip.
        self.DIAG_TOPICS = self.get_parameter('diag_topics').value
        if self.DIAG_TOPICS:
            self.diag_pub_n_eff = self.create_publisher(Float32, '/pf/debug/n_eff', 1)
            self.diag_pub_weight_entropy = self.create_publisher(Float32, '/pf/debug/weight_entropy', 1)
            self.diag_pub_pose_cov_trace = self.create_publisher(Float32, '/pf/debug/pose_cov_trace', 1)
            self.diag_pub_update_hz = self.create_publisher(Float32, '/pf/debug/update_hz', 1)
            self.diag_pub_frac_clamped = self.create_publisher(Float32, '/pf/debug/frac_clamped', 1)
            self.diag_pub_frac_short = self.create_publisher(Float32, '/pf/debug/frac_short', 1)
            self.get_logger().info('Diagnostic scalar topics enabled on /pf/debug/*')

        # live likelihood-field debug grid (Phase 3d Task 4): disabled by
        # default (no publisher created, no extra raycasts, zero
        # measurable overhead). When enabled, at most every LF_PERIOD_S
        # seconds, builds a coarse pose grid centred on the current
        # inferred pose and publishes it as a nav_msgs/OccupancyGrid on
        # /pf/debug/likelihood_field for RViz overlay -- see
        # build_likelihood_field(). This is purely a diagnostic
        # visualisation of the sensor model's likelihood surface (see
        # docs/research/localization/2d_mcl_algorithm.md sec 5.4, the
        # beam-correlation "ridge" this is meant to expose); it does not
        # feed back into the particle filter state.
        self.LF_ENABLE     = self.get_parameter('likelihood_field_enable').value
        self.LF_WINDOW_M   = self.get_parameter('lf_window_m').value
        self.LF_RES_M      = self.get_parameter('lf_res_m').value
        self.LF_PERIOD_S   = self.get_parameter('lf_period_s').value
        self.LF_LOG_FLOOR  = self.get_parameter('lf_log_floor').value
        self._lf_last_wall = None
        self._lf_cache = {}
        if self.LF_ENABLE:
            self.lf_pub = self.create_publisher(OccupancyGrid, '/pf/debug/likelihood_field', 1)
            self.get_logger().info('Likelihood-field debug grid enabled on /pf/debug/likelihood_field')

        # sensor model constants
        self.Z_SHORT   = self.get_parameter('z_short').value
        self.Z_MAX     = self.get_parameter('z_max').value
        self.Z_RAND    = self.get_parameter('z_rand').value
        self.Z_HIT     = self.get_parameter('z_hit').value
        self.SIGMA_HIT = self.get_parameter('sigma_hit').value
        # Phase 3e Task 2: sensor-model variant selection (see
        # particle_filter/sensor_model.py and
        # docs/research/localization/2d_mcl_algorithm.md sec 5.1). Default
        # ('upstream') preserves today's (unnormalised p_short) behavior
        # exactly. 'normalized_short' normalises the short-reading
        # component per column so the configured z_short weight means what
        # it says at every predicted range; SENSOR_MODEL_LAMBDA_SHORT (1/px)
        # only affects that variant.
        self.SENSOR_MODEL_VARIANT      = self.get_parameter('sensor_model_variant').value
        self.SENSOR_MODEL_LAMBDA_SHORT = self.get_parameter('sensor_model_lambda_short').value

        # Phase 3e Task 3: drop non-finite (no-return) observed beams from
        # sensor-model evaluation entirely, instead of letting them fall
        # into the max-range table bucket where they over-reward particles
        # (see docs/research/localization/2d_mcl_algorithm.md sec 5.2).
        # Default (False) preserves today's behavior exactly -- see
        # sensor_model()/_eval_sensor_model_skip_nonfinite() below.
        self.SKIP_NONFINITE_BEAMS = self.get_parameter('skip_nonfinite_beams').value

        # Phase 3e Task 3 robustness guard (flagged during Task 3 review):
        # if masking non-finite beams leaves fewer than MIN_FINITE_BEAMS
        # surviving beams, skip the correction for this update (uniform
        # weights, same fallback as the n_finite==0 case) instead of
        # evaluating an empty or near-empty beam set, which would let a
        # handful of beams dominate the whole update. Only reachable when
        # SKIP_NONFINITE_BEAMS is True (n_finite is only computed on that
        # path) -- with the default skip_nonfinite_beams=False this
        # parameter is inert and cannot change behavior.
        self.MIN_FINITE_BEAMS = self.get_parameter('min_finite_beams').value

        # Phase 3e Task 4 (see docs/research/localization/2d_mcl_algorithm.md
        # sec 5.3): odomCB fires at odometry rate (~20 Hz) while scans
        # arrive at ~10 Hz, so upstream's "correct on every odomCB" fires
        # the MCL correction twice per scan, roughly squaring the
        # per-scan likelihood contribution. When True, `update()` runs
        # the full correction only when a not-yet-consumed scan is
        # available (see should_run_correction()); odometry deltas
        # accumulate across the skipped odomCB calls via
        # compose_odometry_delta() (exact rotation composition, not an
        # approximation -- see that function's docstring). Default
        # (False) preserves upstream behavior exactly.
        self.UPDATE_ON_NEW_SCAN_ONLY = self.get_parameter(
            'update_on_new_scan_only').value
        self.last_scan_stamp = None
        self._last_corrected_scan_stamp = None

        # motion model constants
        self.MOTION_DISPERSION_X     = self.get_parameter('motion_dispersion_x').value
        self.MOTION_DISPERSION_Y     = self.get_parameter('motion_dispersion_y').value
        self.MOTION_DISPERSION_THETA = self.get_parameter('motion_dispersion_theta').value
        
        # various data containers used in the MCL algorithm
        self.MAX_RANGE_PX = None
        self.odometry_data = np.array([0.0, 0.0, 0.0])
        self.laser = None
        self.iters = 0
        self.map_info = None
        self.map_initialized = False
        self.lidar_initialized = False
        self.odom_initialized = False
        self.last_pose = None
        self.laser_angles = None
        self.downsampled_angles = None
        self.range_method = None
        self.last_time = None
        self.last_stamp = None
        self.first_sensor_update = True
        self.state_lock = Lock()

        # cache this to avoid memory allocation in motion model
        self.local_deltas = np.zeros((self.MAX_PARTICLES, 3))

        # cache this for the sensor model computation
        self.queries = None
        self.ranges = None
        self.tiled_angles = None
        self.sensor_model_table = None

        # particle poses and weights
        self.inferred_pose = None
        self.particle_indices = np.arange(self.MAX_PARTICLES)
        self.particles = np.zeros((self.MAX_PARTICLES, 3))
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)

        # initialize the state
        self.smoothing = Utils.CircularArray(10)
        self.timer = Utils.Timer(10)
        # map service client
        self.map_client = self.create_client(GetMap, '/map_server/map')
        self.get_omap()
        self.precompute_sensor_model()
        self.initialize_global()

        # keep track of speed from input odom
        self.current_speed = 0.0

        # Pub Subs
        # these topics are for visualization
        self.pose_pub = self.create_publisher(PoseStamped, '/pf/viz/inferred_pose', 1)
        self.particle_pub = self.create_publisher(PoseArray, '/pf/viz/particles', 1)
        self.pub_fake_scan = self.create_publisher(LaserScan, '/pf/viz/fake_scan', 1)
        self.rect_pub = self.create_publisher(PolygonStamped, '/pf/viz/poly1', 1)

        if self.PUBLISH_ODOM:
            self.odom_pub = self.create_publisher(Odometry, '/pf/pose/odom', 1)

        # these topics are for coordinate space things
        self.pub_tf = TransformBroadcaster(self)

        # these topics are to receive data from the racecar
        self.laser_sub = self.create_subscription(
            LaserScan,
            self.get_parameter('scan_topic').value,
            self.lidarCB,
            1)
        self.odom_sub = self.create_subscription(
            Odometry,
            self.get_parameter('odometry_topic').value,
            self.odomCB,
            1)
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped,
            '/initialpose',
            self.clicked_pose,
            1)
        self.click_sub = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_pose,
            1)

        self.get_logger().info('Finished initializing, waiting on messages...')

    def get_omap(self):
        '''
        Fetch the occupancy grid map from the map_server instance, and initialize the correct
        RangeLibc method. Also stores a matrix which indicates the permissible region of the map
        '''

        while not self.map_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Get map service not available, waiting...')
        req = GetMap.Request()
        future = self.map_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        map_msg = future.result().map
        self.map_info = map_msg.info

        oMap = range_libc.PyOMap(map_msg)
        self.MAX_RANGE_PX = int(self.MAX_RANGE_METERS / self.map_info.resolution)

        # initialize range method
        self.get_logger().info('Initializing range method: ' + self.WHICH_RM)
        if self.WHICH_RM == 'bl':
            self.range_method = range_libc.PyBresenhamsLine(oMap, self.MAX_RANGE_PX)
        elif 'cddt' in self.WHICH_RM:
            self.range_method = range_libc.PyCDDTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
            if self.WHICH_RM == 'pcddt':
                self.get_logger().info('Pruning...')
                self.range_method.prune()
        elif self.WHICH_RM == 'rm':
            self.range_method = range_libc.PyRayMarching(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'rmgpu':
            self.range_method = range_libc.PyRayMarchingGPU(oMap, self.MAX_RANGE_PX)
        elif self.WHICH_RM == 'glt':
            self.range_method = range_libc.PyGiantLUTCast(oMap, self.MAX_RANGE_PX, self.THETA_DISCRETIZATION)
        self.get_logger().info('Done loading map')

         # 0: permissible, -1: unmapped, 100: blocked
        array_255 = np.array(map_msg.data).reshape((map_msg.info.height, map_msg.info.width))

        # 0: not permissible, 1: permissible
        self.permissible_region = np.zeros_like(array_255, dtype=bool)
        self.permissible_region[array_255==0] = 1
        self.map_initialized = True

    def publish_tf(self, pose, stamp=None):
        ''' Publish a tf for the car. This tells ROS where the car is with respect to the map. '''
        if stamp == None:
            stamp = self.get_clock().now().to_msg()

        t = TransformStamped()
        # header
        t.header.stamp = stamp
        t.header.frame_id = '/map'
        t.child_frame_id = '/laser'
        # translation
        t.transform.translation.x = pose[0]
        t.transform.translation.y = pose[1]
        t.transform.translation.z = 0.0
        q = tf_transformations.quaternion_from_euler(0., 0., pose[2])
        # rotation
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        self.pub_tf.sendTransform(t)
        # also publish odometry to facilitate getting the localization pose
        if self.PUBLISH_ODOM:
            odom = Odometry()
            odom.header.stamp = self.get_clock().now().to_msg()
            odom.header.frame_id = '/map'
            odom.pose.pose.position.x = pose[0]
            odom.pose.pose.position.y = pose[1]
            odom.pose.pose.orientation = Utils.angle_to_quaternion(pose[2])
            cov_mat = np.cov(self.particles, rowvar=False, ddof=0, aweights=self.weights).flatten()
            odom.pose.covariance[:cov_mat.shape[0]] = cov_mat
            odom.twist.twist.linear.x = self.current_speed
            self.odom_pub.publish(odom)
        
        return

    def visualize(self):
        '''
        Publish various visualization messages.
        '''
        if not self.DO_VIZ:
            return

        if self.pose_pub.get_subscription_count() > 0 and isinstance(self.inferred_pose, np.ndarray):
            # Publish the inferred pose for visualization
            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = '/map'
            ps.pose.position.x = self.inferred_pose[0]
            ps.pose.position.y = self.inferred_pose[1]
            ps.pose.orientation = Utils.angle_to_quaternion(self.inferred_pose[2])
            self.pose_pub.publish(ps)

        if self.particle_pub.get_subscription_count() > 0:
            # publish a downsampled version of the particle distribution to avoid a lot of latency
            if self.MAX_PARTICLES > self.MAX_VIZ_PARTICLES:
                # randomly downsample particles
                proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES, p=self.weights)
                # proposal_indices = np.random.choice(self.particle_indices, self.MAX_VIZ_PARTICLES)
                self.publish_particles(self.particles[proposal_indices,:])
            else:
                self.publish_particles(self.particles)

        if self.pub_fake_scan.get_subscription_count() > 0 and isinstance(self.ranges, np.ndarray):
            # generate the scan from the point of view of the inferred position for visualization
            self.viz_queries[:,0] = self.inferred_pose[0]
            self.viz_queries[:,1] = self.inferred_pose[1]
            self.viz_queries[:,2] = self.downsampled_angles + self.inferred_pose[2]
            self.range_method.calc_range_many(self.viz_queries, self.viz_ranges)
            self.publish_scan(self.downsampled_angles, self.viz_ranges)

    def publish_particles(self, particles):
        # publish the given particles as a PoseArray object
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = '/map'
        pa.poses = Utils.particles_to_poses(particles)
        self.particle_pub.publish(pa)

    def publish_scan(self, angles, ranges):
        # publish the given angels and ranges as a laser scan message
        ls = LaserScan()
        ls.header.stamp = self.last_stamp
        ls.header.frame_id = '/laser'
        ls.angle_min = np.min(angles)
        ls.angle_max = np.max(angles)
        ls.angle_increment = np.abs(angles[0] - angles[1])
        ls.range_min = 0
        ls.range_max = np.max(ranges)
        ls.ranges = ranges
        self.pub_fake_scan.publish(ls)

    def lidarCB(self, msg):
        '''
        Initializes reused buffers, and stores the relevant laser scanner data for later use.
        '''
        if not isinstance(self.laser_angles, np.ndarray):
            self.get_logger().info('...Received first LiDAR message')
            self.laser_angles = np.linspace(msg.angle_min, msg.angle_max, len(msg.ranges))
            self.downsampled_angles = np.copy(self.laser_angles[0::self.ANGLE_STEP]).astype(np.float32)
            self.viz_queries = np.zeros((self.downsampled_angles.shape[0],3), dtype=np.float32)
            self.viz_ranges = np.zeros(self.downsampled_angles.shape[0], dtype=np.float32)
            self.get_logger().info(str(self.downsampled_angles.shape[0]))

        # store the necessary scanner information for later processing
        self.downsampled_ranges = np.array(msg.ranges[::self.ANGLE_STEP])
        self.lidar_initialized = True
        # Phase 3e Task 4: record this scan's stamp so update() can tell
        # (via should_run_correction()) whether a not-yet-consumed scan is
        # available when update_on_new_scan_only is set. header.stamp is a
        # builtin_interfaces/Time (sec, nanosec); store the tuple so
        # equality comparison doesn't depend on message-object identity.
        self.last_scan_stamp = (
            msg.header.stamp.sec, msg.header.stamp.nanosec)
        # self.update()

    def odomCB(self, msg):
        '''
        Store deltas between consecutive odometry messages in the coordinate space of the car.

        Odometry data is accumulated via dead reckoning, so it is very inaccurate on its own.
        '''
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y])

        orientation = Utils.quaternion_to_angle(msg.pose.pose.orientation)
        pose = np.array([position[0], position[1], orientation])
        self.current_speed = msg.twist.twist.linear.x

        if isinstance(self.last_pose, np.ndarray):
            # changes in x,y,theta in local coordinate system of the car
            rot = Utils.rotation_matrix(-self.last_pose[2])
            delta = np.array([position - self.last_pose[0:2]]).transpose()
            local_delta = (rot*delta).transpose()

            # Phase 3e Task 4: fold this step's local-frame delta into the
            # running accumulator instead of overwriting it, so deltas
            # aren't lost when update() skips a correction (gated path).
            # This is a strict generalization of the old overwrite: when
            # self.odometry_data is [0,0,0] (always true here in the
            # default/ungated path, since update() zeroes it after every
            # single odomCB-triggered call), compose_odometry_delta()
            # reduces to the old assignment exactly -- see its docstring.
            step_delta = np.array([
                local_delta[0, 0], local_delta[0, 1],
                orientation - self.last_pose[2]])
            self.odometry_data = compose_odometry_delta(
                self.odometry_data, step_delta)
            self.last_pose = pose
            self.last_stamp = msg.header.stamp
            self.odom_initialized = True
        else:
            self.get_logger().info('...Received first Odometry message')
            self.last_pose = pose

        # this topic is slower than lidar, so update every time we receive a message
        self.update()

    def clicked_pose(self, msg):
        '''
        Receive pose messages from RViz and initialize the particle distribution in response.
        '''
        if isinstance(msg, PointStamped):
            self.initialize_global()
        elif isinstance(msg, PoseWithCovarianceStamped):
            self.initialize_particles_pose(msg.pose.pose)

    def initialize_particles_pose(self, pose):
        '''
        Initialize particles in the general region of the provided pose.
        '''
        self.get_logger().info('SETTING POSE')
        self.get_logger().info(str([pose.position.x, pose.position.y]))
        self.state_lock.acquire()
        self.weights = np.ones(self.MAX_PARTICLES) / float(self.MAX_PARTICLES)
        self.particles[:,0] = pose.position.x + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
        self.particles[:,1] = pose.position.y + np.random.normal(loc=0.0,scale=0.5,size=self.MAX_PARTICLES)
        self.particles[:,2] = Utils.quaternion_to_angle(pose.orientation) + np.random.normal(loc=0.0,scale=0.4,size=self.MAX_PARTICLES)
        self.state_lock.release()

    def initialize_global(self):
        '''
        Spread the particle distribution over the permissible region of the state space.
        '''
        self.get_logger().info('GLOBAL INITIALIZATION')
        # randomize over grid coordinate space
        self.state_lock.acquire()
        permissible_x, permissible_y = np.where(self.permissible_region == 1)
        indices = np.random.randint(0, len(permissible_x), size=self.MAX_PARTICLES)

        permissible_states = np.zeros((self.MAX_PARTICLES,3))
        permissible_states[:,0] = permissible_y[indices]
        permissible_states[:,1] = permissible_x[indices]
        permissible_states[:,2] = np.random.random(self.MAX_PARTICLES) * np.pi * 2.0

        Utils.map_to_world(permissible_states, self.map_info)
        self.particles = permissible_states
        self.weights[:] = 1.0 / self.MAX_PARTICLES
        self.state_lock.release()

    def precompute_sensor_model(self):
        '''
        Generate and store a table which represents the sensor model. For each discrete computed
        range value, this provides the probability of measuring any (discrete) range.

        This table is indexed by the sensor model at runtime by discretizing the measurements
        and computed ranges from RangeLibc.
        '''
        self.get_logger().info('Precomputing sensor model')
        # sensor model constants
        z_short = self.Z_SHORT
        z_max   = self.Z_MAX
        z_rand  = self.Z_RAND
        z_hit   = self.Z_HIT
        sigma_hit = self.SIGMA_HIT

        t = time.time()
        self.sensor_model_table = build_table(
            self.MAX_RANGE_PX, z_hit, z_short, z_max, z_rand, sigma_hit,
            variant=self.SENSOR_MODEL_VARIANT,
            lambda_short=self.SENSOR_MODEL_LAMBDA_SHORT,
        )

        # upload the sensor model to RangeLib for ultra fast resolution
        if self.RANGELIB_VAR > 0:
            self.range_method.set_sensor_model(self.sensor_model_table)

    def motion_model(self, proposal_dist, action):
        '''
        The motion model applies the odometry to the particle distribution. Since there the odometry
        data is inaccurate, the motion model mixes in gaussian noise to spread out the distribution.

        Vectorized motion model. Computing the motion model over all particles is thousands of times
        faster than doing it for each particle individually due to vectorization and reduction in
        function call overhead
        
        TODO this could be better, but it works for now
            - fixed random noise is not very realistic
            - ackermann model provides bad estimates at high speed
        '''
        # rotate the action into the coordinate space of each particle
        # t1 = time.time()
        cosines = np.cos(proposal_dist[:,2])
        sines = np.sin(proposal_dist[:,2])

        self.local_deltas[:,0] = cosines*action[0] - sines*action[1]
        self.local_deltas[:,1] = sines*action[0] + cosines*action[1]
        self.local_deltas[:,2] = action[2]

        proposal_dist[:,:] += self.local_deltas
        proposal_dist[:,0] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_X,size=self.MAX_PARTICLES)
        proposal_dist[:,1] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_Y,size=self.MAX_PARTICLES)
        proposal_dist[:,2] += np.random.normal(loc=0.0,scale=self.MOTION_DISPERSION_THETA,size=self.MAX_PARTICLES)

    def _eval_sensor_model_skip_nonfinite(self, obs, num_rays):
        '''
        Phase 3e Task 3: evaluate the sensor model excluding non-finite
        (no-return) observed beams -- and their corresponding predicted
        ranges -- from the per-particle likelihood product entirely,
        for every particle. This matches
        scripts/2dlidar/score_sensor_model.py's
        apply_skip_nonfinite_mask() semantics exactly (same beams
        dropped, same pairing of observed range <-> angle/predicted
        range), so the offline gate score is a truthful prediction of
        this online behavior -- not merely "measurably equivalent" via a
        neutral-multiplier correction.

        Only called from sensor_model() when SKIP_NONFINITE_BEAMS is
        True (opt-in); the default (False) path calls
        range_method.eval_sensor_model() directly against the full,
        fixed-size self.ranges buffer, unchanged -- so that fast path's
        arithmetic is untouched by this method existing.

        self.ranges is laid out particle-major (ranges[i*num_rays+j]),
        filled by calc_range_repeat_angles()/calc_range_many() just
        before this is called -- true for both
        VAR_REPEAT_ANGLES_EVAL_SENSOR and VAR_CALC_RANGE_MANY_EVAL_SENSOR,
        so this helper is shared by both.

        A fresh (n_finite-sized) obs/ranges buffer is allocated every
        call rather than reusing a preallocated one, since n_finite
        varies scan-to-scan; per the task brief this is acceptable
        because eval_sensor_model is only a fraction of one update's
        cost (the ray casting that fills self.ranges is unchanged/still
        full-size). The masking itself is `select_finite_beams` (module
        level, pure, unit-tested independently of ROS/range_libc).
        '''
        ranges_2d = self.ranges[:num_rays * self.MAX_PARTICLES].reshape(
            self.MAX_PARTICLES, num_rays)
        obs_finite, ranges_finite, n_finite = select_finite_beams(obs, ranges_2d)
        if n_finite < self.MIN_FINITE_BEAMS:
            # Fewer than MIN_FINITE_BEAMS usable beams this update (Phase
            # 3e Task 3 robustness guard, min_finite_beams param, default
            # 10): no reliable information, so every particle is equally
            # (un)likely -- same fallback as the degenerate n_finite==0
            # case (which is subsumed here since 0 < MIN_FINITE_BEAMS for
            # any sane threshold). Short-circuit explicitly instead of
            # calling into eval_sensor_model with an empty or near-empty
            # buffer, since a handful of surviving beams would otherwise
            # dominate the whole update, and a zero-length buffer would
            # hit the pybind wrapper's unconditional obs[0]/ranges[0]
            # dereference before the C++ loop even runs.
            self.weights[:] = 1.0
            return

        self.range_method.eval_sensor_model(
            obs_finite, ranges_finite, self.weights, n_finite, self.MAX_PARTICLES)

    def sensor_model(self, proposal_dist, obs, weights):
        '''
        This function computes a probablistic weight for each particle in the proposal distribution.
        These weights represent how probable each proposed (x,y,theta) pose is given the measured
        ranges from the lidar scanner.

        There are 4 different variants using various features of RangeLibc for demonstration purposes.
        - VAR_REPEAT_ANGLES_EVAL_SENSOR is the most stable, and is very fast.
        - VAR_NO_EVAL_SENSOR_MODEL directly indexes the precomputed sensor model. This is slow
                                   but it demonstrates what self.range_method.eval_sensor_model does
        - VAR_RADIAL_CDDT_OPTIMIZATIONS is only compatible with CDDT or PCDDT, it implments the radial
                                        optimizations to CDDT which simultaneously performs ray casting
                                        in two directions, reducing the amount of work by roughly a third
        '''
        
        num_rays = self.downsampled_angles.shape[0]
        # only allocate buffers once to avoid slowness
        if self.first_sensor_update:
            if self.RANGELIB_VAR <= 1:
                self.queries = np.zeros((num_rays*self.MAX_PARTICLES,3), dtype=np.float32)
            else:
                self.queries = np.zeros((self.MAX_PARTICLES,3), dtype=np.float32)

            self.ranges = np.zeros(num_rays*self.MAX_PARTICLES, dtype=np.float32)
            self.tiled_angles = np.tile(self.downsampled_angles, self.MAX_PARTICLES)
            self.first_sensor_update = False

        if self.RANGELIB_VAR == VAR_RADIAL_CDDT_OPTIMIZATIONS:
            if 'cddt' in self.WHICH_RM:
                self.queries[:,:] = proposal_dist[:,:]
                self.range_method.calc_range_many_radial_optimized(num_rays, self.downsampled_angles[0], self.downsampled_angles[-1], self.queries, self.ranges)

                # evaluate the sensor model
                self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
                # apply the squash factor
                self.weights = np.power(self.weights, self.INV_SQUASH_FACTOR)
            else:
                self.get_logger().info('Cannot use radial optimizations with non-CDDT based methods, use rangelib_variant 2')
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR_ONE_SHOT:
            self.queries[:,:] = proposal_dist[:,:]
            self.range_method.calc_range_repeat_angles_eval_sensor_model(self.queries, self.downsampled_angles, obs, self.weights)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_REPEAT_ANGLES_EVAL_SENSOR:
            if self.SHOW_FINE_TIMING:
                t_start = time.time()
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            self.queries[:,:] = proposal_dist[:,:]
            if self.SHOW_FINE_TIMING:
                t_init = time.time()
            self.range_method.calc_range_repeat_angles(self.queries, self.downsampled_angles, self.ranges)
            if self.SHOW_FINE_TIMING:
                t_range = time.time()
            # evaluate the sensor model on the GPU
            if self.SKIP_NONFINITE_BEAMS:
                self._eval_sensor_model_skip_nonfinite(obs, num_rays)
            else:
                self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            if self.SHOW_FINE_TIMING:
                t_eval = time.time()
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
            if self.SHOW_FINE_TIMING:
                t_squash = time.time()
                t_total = (t_squash - t_start) / 100.0

            if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
                self.get_logger().info(str(['sensor_model: init: ', np.round((t_init-t_start)/t_total, 2), 'range:', np.round((t_range-t_init)/t_total, 2), \
                      'eval:', np.round((t_eval-t_range)/t_total, 2), 'squash:', np.round((t_squash-t_eval)/t_total, 2)]))
        elif self.RANGELIB_VAR == VAR_CALC_RANGE_MANY_EVAL_SENSOR:
            # this version demonstrates what this would look like with coordinate space conversion pushed to rangelib
            # this part is inefficient since it requires a lot of effort to construct this redundant array
            self.queries[:,0] = np.repeat(proposal_dist[:,0], num_rays)
            self.queries[:,1] = np.repeat(proposal_dist[:,1], num_rays)
            self.queries[:,2] = np.repeat(proposal_dist[:,2], num_rays)
            self.queries[:,2] += self.tiled_angles

            self.range_method.calc_range_many(self.queries, self.ranges)

            # evaluate the sensor model on the GPU
            if self.SKIP_NONFINITE_BEAMS:
                self._eval_sensor_model_skip_nonfinite(obs, num_rays)
            else:
                self.range_method.eval_sensor_model(obs, self.ranges, self.weights, num_rays, self.MAX_PARTICLES)
            np.power(self.weights, self.INV_SQUASH_FACTOR, self.weights)
        elif self.RANGELIB_VAR == VAR_NO_EVAL_SENSOR_MODEL:
            # this version directly uses the sensor model in Python, at a significant computational cost
            self.queries[:,0] = np.repeat(proposal_dist[:,0], num_rays)
            self.queries[:,1] = np.repeat(proposal_dist[:,1], num_rays)
            self.queries[:,2] = np.repeat(proposal_dist[:,2], num_rays)
            self.queries[:,2] += self.tiled_angles

            # compute the ranges for all the particles in a single functon call
            self.range_method.calc_range_many(self.queries, self.ranges)

            # resolve the sensor model by discretizing and indexing into the precomputed table
            obs /= float(self.map_info.resolution)
            ranges = self.ranges / float(self.map_info.resolution)
            obs[obs > self.MAX_RANGE_PX] = self.MAX_RANGE_PX
            ranges[ranges > self.MAX_RANGE_PX] = self.MAX_RANGE_PX

            intobs = np.rint(obs).astype(np.uint16)
            intrng = np.rint(ranges).astype(np.uint16)

            # compute the weight for each particle
            for i in range(self.MAX_PARTICLES):
                weight = np.product(self.sensor_model_table[intobs,intrng[i*num_rays:(i+1)*num_rays]])
                weight = np.power(weight, self.INV_SQUASH_FACTOR)
                weights[i] = weight
        else:
            self.get_logger().info('PLEASE SET rangelib_variant PARAM to 0-4')

    def build_likelihood_field(self, obs):
        '''
        Phase 3d Task 4: build a coarse pose grid centred on the current
        inferred pose (lf_window_m across, spacing lf_res_m, heading
        fixed at the inferred theta), evaluate it through the SAME
        calc_range_repeat_angles + eval_sensor_model path used by
        sensor_model()'s VAR_REPEAT_ANGLES_EVAL_SENSOR variant against
        the current downsampled scan, and publish the resulting
        likelihood surface as a nav_msgs/OccupancyGrid on
        /pf/debug/likelihood_field.

        Encoding: log(weight), minus the max (best pose -> 0), clipped
        at -lf_log_floor, mapped linearly to 0..100 with 100 = most
        likely -- so RViz's costmap colour scheme lights up the ridge/
        peak of the likelihood surface.

        Rate-limited to at most once every lf_period_s seconds (wall
        clock). Uses its own preallocated buffers, cached by grid shape
        -- NOT the MAX_PARTICLES-sized buffers from sensor_model(),
        since the pose-grid size (n*n) is independent of MAX_PARTICLES.
        '''
        if not isinstance(self.inferred_pose, np.ndarray) or not isinstance(self.downsampled_angles, np.ndarray):
            return

        now = time.time()
        if self._lf_last_wall is not None and (now - self._lf_last_wall) < self.LF_PERIOD_S:
            return
        self._lf_last_wall = now

        n = int(round(self.LF_WINDOW_M / self.LF_RES_M)) + 1
        num_rays = self.downsampled_angles.shape[0]
        num_poses = n * n

        # cache preallocated buffers by (grid_side, num_rays) shape
        cache_key = (n, num_rays)
        cached = self._lf_cache.get(cache_key)
        if cached is None:
            cached = (
                np.zeros((num_poses, 3), dtype=np.float32),      # lf_queries
                np.zeros(num_poses * num_rays, dtype=np.float32),  # lf_ranges
                np.zeros(num_poses, dtype=np.float64),           # lf_weights
            )
            self._lf_cache[cache_key] = cached
        lf_queries, lf_ranges, lf_weights = cached

        center_x = float(self.inferred_pose[0])
        center_y = float(self.inferred_pose[1])
        theta = float(self.inferred_pose[2])

        offsets = (np.arange(n) - (n - 1) / 2.0) * self.LF_RES_M
        xs = center_x + offsets
        ys = center_y + offsets

        # row-major grid: column index advances with x, row index
        # advances with y -- matches nav_msgs/OccupancyGrid's
        # data[width*row + col] layout when info.origin is placed at
        # the (x,y) of the first cell's corner (see below).
        grid_x, grid_y = np.meshgrid(xs, ys)
        lf_queries[:, 0] = grid_x.ravel().astype(np.float32)
        lf_queries[:, 1] = grid_y.ravel().astype(np.float32)
        lf_queries[:, 2] = np.float32(theta)

        # SAME raycast + eval path as sensor_model()'s
        # VAR_REPEAT_ANGLES_EVAL_SENSOR variant, against separate buffers.
        self.range_method.calc_range_repeat_angles(lf_queries, self.downsampled_angles, lf_ranges)
        self.range_method.eval_sensor_model(obs, lf_ranges, lf_weights, num_rays, num_poses)

        with np.errstate(divide='ignore'):
            log_w = np.log(lf_weights)
        log_w = log_w - np.max(log_w)
        log_w = np.clip(log_w, -self.LF_LOG_FLOOR, 0.0)
        occ = np.rint((log_w + self.LF_LOG_FLOOR) * (100.0 / self.LF_LOG_FLOOR)).astype(np.int8)

        grid_msg = OccupancyGrid()
        grid_msg.header.stamp = self.get_clock().now().to_msg()
        grid_msg.header.frame_id = 'map'
        grid_msg.info.resolution = float(self.LF_RES_M)
        grid_msg.info.width = n
        grid_msg.info.height = n
        # info.origin is the pose of the *corner* of cell (0,0); xs[0]/
        # ys[0] are cell CENTRES, so shift back by half a cell.
        grid_msg.info.origin.position.x = float(xs[0] - self.LF_RES_M / 2.0)
        grid_msg.info.origin.position.y = float(ys[0] - self.LF_RES_M / 2.0)
        grid_msg.info.origin.position.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0
        grid_msg.data = occ.reshape(-1).tolist()
        self.lf_pub.publish(grid_msg)

    def MCL(self, a, o):
        '''
        Performs one step of Monte Carlo Localization.
            1. resample particle distribution to form the proposal distribution
            2. apply the motion model
            3. apply the sensor model
            4. normalize particle weights

        This is in the critical path of code execution, so it is optimized for speed.
        '''
        # Diagnostics (Phase 3d Task 1) reuses these same timing points --
        # compute them unconditionally when diagnostics are enabled so
        # there is exactly one timing path (not a second one alongside
        # SHOW_FINE_TIMING).
        need_timing = self.SHOW_FINE_TIMING or self.DIAG_ENABLE or self.DIAG_TOPICS
        if need_timing:
            t = time.time()

        # ESS gate (Phase 3c Lever 3): decide whether to resample this
        # update. Disabled (USE_ESS_GATE=False) reproduces upstream
        # behavior exactly -- always resample every update.
        if self.USE_ESS_GATE:
            do_resample = should_resample(self.weights, self.MAX_PARTICLES, self.ESS_THRESHOLD_RATIO)
        else:
            do_resample = True

        if do_resample:
            # draw the proposal distribution from the old particles
            proposal_indices = np.random.choice(self.particle_indices, self.MAX_PARTICLES, p=self.weights)
            proposal_distribution = self.particles[proposal_indices,:]
            prior_weights = None
        else:
            # carry particles forward unchanged (no resampling this update)
            proposal_distribution = np.copy(self.particles)
            # sensor_model() below overwrites self.weights with the raw
            # per-particle likelihood for this update; save the prior
            # (post-normalization) weights so they can be folded back in
            # as an importance-weight multiply, keeping this branch a
            # mathematically correct sequential-importance-sampling step
            # (weight_new = weight_prior * likelihood) instead of
            # discarding the prior weight distribution.
            prior_weights = np.copy(self.weights)
        if need_timing:
            t_propose = time.time()

        # compute the motion model to update the proposal distribution
        self.motion_model(proposal_distribution, a)
        if need_timing:
            t_motion = time.time()

        # compute the sensor model
        self.sensor_model(proposal_distribution, o, self.weights)
        if need_timing:
            t_sensor = time.time()

        if prior_weights is not None:
            # no-resample path: fold the prior weight distribution back in
            # (see comment above) before normalizing.
            self.weights *= prior_weights

        # normalize importance weights
        self.weights /= np.sum(self.weights)
        if need_timing:
            t_norm = time.time()

        if self.SHOW_FINE_TIMING and self.iters % 10 == 0:
            t_total = (t_norm - t)/100.0
            self.get_logger().info(str(['MCL: propose: ', np.round((t_propose-t)/t_total, 2), 'motion:', np.round((t_motion-t_propose)/t_total, 2), \
                  'sensor:', np.round((t_sensor-t_motion)/t_total, 2), 'norm:', np.round((t_norm-t_sensor)/t_total, 2)]))

        # save the particles
        self.particles = proposal_distribution

        if self.DIAG_ENABLE or self.DIAG_TOPICS:
            self.record_diagnostics(a, o, do_resample, t, t_propose, t_motion, t_sensor, t_norm)

    def record_diagnostics(self, a, o, do_resample, t, t_propose, t_motion, t_sensor, t_norm):
        '''
        Build one diagnostics record (Phase 3d Task 1) and, depending on
        which flags are enabled, append it to the JSONL recorder
        (diag_enable) and/or publish a subset of its values as live
        std_msgs/Float32 topics under /pf/debug/* (diag_topics, Phase 3d
        Task 2). Called from MCL() only when at least one of those flags
        is true, so this adds zero overhead in the default (off)
        configuration. JSONL record schema is fixed -- the offline
        plotter (Task 3) depends on these exact keys -- see
        docs/design/f1tenth-2dlidar-integration or the task brief for the
        full schema.
        '''
        if self.iters % self.DIAG_EVERY != 0:
            return

        num_rays = self.downsampled_angles.shape[0] if isinstance(self.downsampled_angles, np.ndarray) else 0
        best_idx = int(np.argmax(self.weights))
        predicted_best = None
        frac = {'hit': 0.0, 'short': 0.0, 'long': 0.0, 'clamped': 0.0, 'nonfinite': 0.0}
        if num_rays > 0 and isinstance(self.ranges, np.ndarray) and \
                self.ranges.shape[0] >= (best_idx + 1) * num_rays:
            start = best_idx * num_rays
            predicted_best = self.ranges[start:start + num_rays]
            frac = beam_categories(o, predicted_best, self.map_info.resolution,
                                    self.MAX_RANGE_PX, self.SIGMA_HIT)

        cov = pose_covariance(self.particles, self.weights)
        pose = self.inferred_pose if isinstance(self.inferred_pose, np.ndarray) else self.expected_pose()

        now_wall = time.time()
        stamp_scan = None
        if self.last_stamp is not None:
            stamp_scan = self.last_stamp.sec + self.last_stamp.nanosec * 1e-9
        dt_update = 0.0 if self._diag_last_wall is None else (now_wall - self._diag_last_wall)
        self._diag_last_wall = now_wall

        record = {
            'iter': int(self.iters),
            'stamp_scan': stamp_scan,
            'stamp_wall': now_wall,
            'dt_update': dt_update,
            'action_dx': float(a[0]),
            'action_dy': float(a[1]),
            'action_dtheta': float(a[2]),
            'n_eff': float(effective_sample_size(self.weights)),
            'weight_entropy': float(weight_entropy(self.weights)),
            'weight_max': float(np.max(self.weights)),
            'pose_x': float(pose[0]),
            'pose_y': float(pose[1]),
            'pose_theta': float(pose[2]),
            'cov_xx': float(cov[0, 0]),
            'cov_yy': float(cov[1, 1]),
            'cov_xy': float(cov[0, 1]),
            'resampled': bool(do_resample),
            'frac_hit': frac['hit'],
            'frac_short': frac['short'],
            'frac_long': frac['long'],
            'frac_clamped': frac['clamped'],
            'frac_nonfinite': frac['nonfinite'],
            't_propose': float(t_propose - t),
            't_motion': float(t_motion - t_propose),
            't_sensor': float(t_sensor - t_motion),
            't_norm': float(t_norm - t_sensor),
        }
        if self.DIAG_BEAM_ARRAYS:
            record['observed'] = np.asarray(o, dtype=np.float64).tolist()
            record['predicted_best'] = predicted_best.tolist() if predicted_best is not None else None

        if self.diag_recorder is not None:
            self.diag_recorder.record(record)

        if self.DIAG_TOPICS:
            update_hz = 1.0 / dt_update if dt_update > 0.0 else 0.0
            pose_cov_trace = record['cov_xx'] + record['cov_yy']
            self.diag_pub_n_eff.publish(Float32(data=record['n_eff']))
            self.diag_pub_weight_entropy.publish(Float32(data=record['weight_entropy']))
            self.diag_pub_pose_cov_trace.publish(Float32(data=pose_cov_trace))
            self.diag_pub_update_hz.publish(Float32(data=update_hz))
            self.diag_pub_frac_clamped.publish(Float32(data=record['frac_clamped']))
            self.diag_pub_frac_short.publish(Float32(data=record['frac_short']))

    def expected_pose(self):
        # returns the expected value of the pose given the particle distribution
        return np.dot(self.particles.transpose(), self.weights)

    def update(self):
        '''
        Apply the MCL function to update particle filter state. 

        Ensures the state is correctly initialized, and acquires the state lock before proceeding.
        '''
        if self.lidar_initialized and self.odom_initialized and self.map_initialized:
            # Phase 3e Task 4: when update_on_new_scan_only is set, skip
            # the correction entirely (motion model included -- see the
            # module docstring on should_run_correction()/
            # compose_odometry_delta()) until a not-yet-consumed scan is
            # available. odometry_data keeps accumulating in odomCB across
            # these skipped calls, so no motion is lost -- it is applied
            # in one composed step at the next correction. NOTE: pose/tf
            # publishing (publish_tf, below) only happens when a
            # correction actually runs, so with this flag set the publish
            # rate drops from odom rate (~20 Hz) to scan rate (~10 Hz) --
            # see the Task 4 report for why a separate prediction-only
            # publish path was not added.
            if not should_run_correction(
                    self.UPDATE_ON_NEW_SCAN_ONLY, self.last_scan_stamp,
                    self._last_corrected_scan_stamp):
                return
            if self.state_lock.locked():
                self.get_logger().info('Concurrency error avoided')
            else:
                self._last_corrected_scan_stamp = self.last_scan_stamp
                self.state_lock.acquire()
                self.timer.tick()
                self.iters += 1

                t1 = time.time()
                observation = np.copy(self.downsampled_ranges).astype(np.float32)
                action = np.copy(self.odometry_data)
                self.odometry_data = np.zeros(3)

                # run the MCL update algorithm
                self.MCL(action, observation)

                # compute the expected value of the robot pose
                self.inferred_pose = self.expected_pose()
                self.state_lock.release()
                t2 = time.time()

                # publish transformation frame based on inferred pose
                self.publish_tf(self.inferred_pose, self.last_stamp)

                # this is for tracking particle filter speed
                ips = 1.0 / (t2 - t1)
                self.smoothing.append(ips)
                if self.iters % 10 == 0:
                    self.get_logger().info(str(['iters per sec:', int(self.timer.fps()), ' possible:', int(self.smoothing.mean())]))

                self.visualize()

                if self.LF_ENABLE:
                    self.build_likelihood_field(observation)

# import argparse
# import sys
# parser = argparse.ArgumentParser(description='Particle filter.')
# parser.add_argument('--config', help='Path to yaml file containing config parameters. Helpful for calling node directly with Python for profiling.')

# def load_params_from_yaml(fp):
#     from yaml import load
#     with open(fp, 'r') as infile:
#         yaml_data = load(infile)
#         for param in yaml_data:
#             print 'param:', param, ':', yaml_data[param]
#             rospy.set_param('~'+param, yaml_data[param])

# # this function can be used to generate flame graphs easily
# def make_flamegraph(filterx=None):
#     import flamegraph, os
#     perf_log_path = os.path.join(os.path.dirname(__file__), '../tmp/perf.log')
#     flamegraph.start_profile_thread(fd=open(perf_log_path, 'w'),
#                                     filter=filterx,
#                                     interval=0.001)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFiler()
    try:
        rclpy.spin(pf)
    finally:
        # flush any pending diagnostics records before exiting (Phase 3d
        # Task 1); no-op when diag_enable is false.
        if pf.diag_recorder is not None:
            pf.diag_recorder.close()

if __name__ == '__main__':
    main()

# if __name__=='__main__':
#     rospy.init_node('particle_filter')

#     args,_ = parser.parse_known_args()
#     if args.config:
#         load_params_from_yaml(args.config)

#     # make_flamegraph(r'update')

#     pf = ParticleFiler()
#     rospy.spin()
