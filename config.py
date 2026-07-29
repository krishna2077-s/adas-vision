"""
config.py — All tunable parameters for the ADAS Vision system.

Tweak these values if lane detection is missing lines or picking up noise.
"""

# ---------------------------------------------------------------------------
# Camera / video
# ---------------------------------------------------------------------------
CAMERA_INDEX = 0          # 0 = built-in webcam, 1 = external USB camera
TARGET_FPS   = 30
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
BLUR_KERNEL_SIZE = (5, 5)   # Gaussian blur — larger = smoother but slower
CANNY_LOW        = 50       # Canny edge low threshold
CANNY_HIGH       = 150      # Canny edge high threshold

# ---------------------------------------------------------------------------
# Region of Interest (ROI)
# Trapezoid that masks out sky, bonnet, and roadside clutter.
# Values are fractions of frame height/width (0.0 – 1.0).
#
#   top_left ────── top_right
#      /                  \
#   bottom_left ── bottom_right
#
# Adjust if the ROI cuts off lanes or includes too much noise.
# ---------------------------------------------------------------------------
ROI_TOP_LEFT_X     = 0.42   # Top-left  X of trapezoid
ROI_TOP_RIGHT_X    = 0.58   # Top-right X of trapezoid
ROI_TOP_Y          = 0.60   # Y position of the top edge (60% down the frame)
ROI_BOTTOM_LEFT_X  = 0.05   # Bottom-left  X (near left edge)
ROI_BOTTOM_RIGHT_X = 0.95   # Bottom-right X (near right edge)
ROI_BOTTOM_Y       = 0.95   # Y position of the bottom edge

# ---------------------------------------------------------------------------
# Hough Line Transform
# ---------------------------------------------------------------------------
HOUGH_RHO         = 1       # Distance resolution (pixels)
HOUGH_THETA       = 1       # Angle resolution (degrees, converted internally)
HOUGH_THRESHOLD   = 30      # Minimum votes to consider a line
HOUGH_MIN_LENGTH  = 20      # Minimum line length (pixels)
HOUGH_MAX_GAP     = 200     # Maximum gap between line segments to join them

# Slope thresholds — filters out near-horizontal lines (road markings, not lanes)
MIN_SLOPE = 0.3
MAX_SLOPE = 2.5

# ---------------------------------------------------------------------------
# Lane-marking colour gate + honest failure
# The straight-line detector can only trust actual painted markings. This gate
# keeps only edges sitting on bright white / yellow paint, so kerbs, walls,
# embankments and tree lines can't masquerade as lanes. On a road with no
# usable paint the detector then reports NO LANE (confidence 0.0) and the
# decision engine goes conservative (degraded) — the safe, honest behaviour,
# instead of drawing a confident, wrong lane. Set False for the old edges-only
# mode (marked test tracks only).
# ---------------------------------------------------------------------------
LANE_REQUIRE_MARKINGS = True
WHITE_V_MIN     = 200           # min brightness for 'white' paint (0-255). Swept on real
                                # footage: 200 recovers hazy highway paint (18/25 frames)
                                # while unmarked hill roads still yield 0 false pairs
                                # (geometry + require-both reject the concrete kerb).
WHITE_S_MAX     = 45            # max saturation for white
YELLOW_HSV_LOW  = (18, 90, 150) # HSV lower bound for yellow paint
YELLOW_HSV_HIGH = (38, 255, 255)
MARKING_DILATE  = 11            # px dilation of the paint mask before gating edges

# Geometry sanity — reject an implausible left/right pair (crossed / 'tent'
# lines, or a lane far too wide or narrow) instead of trusting it.
LANE_MIN_BOTTOM_WIDTH_FRAC = 0.20   # min lane width at frame bottom (fraction of width)
LANE_MAX_BOTTOM_WIDTH_FRAC = 0.85   # max lane width at frame bottom
LANE_MIN_TOP_GAP_PX        = 40     # left/right must stay at least this far apart at ROI top

# Require BOTH lanes (a plausible pair) before emitting guidance. A lone edge on
# an unmarked road is almost always a false positive (a kerb or wall), and a
# wrong one-sided steer is dangerous — so a single lane is treated as "no lane"
# (confidence 0.0 -> decision engine goes degraded / conservative). Set False
# only on well-marked roads where one-sided lane-keeping is wanted.
LANE_REQUIRE_BOTH = True

# ---------------------------------------------------------------------------
# MODULE 1b — Drivable-road-surface following (fallback for unmarked roads)
# When paint gives no lane, the road surface itself is detected (smooth +
# grey + connected to the patch ahead of the car) and its curved centreline
# steers the vehicle. Guidance from this path is marked ESTIMATED
# (confidence 0.5) — surface-following is honest but less precise than paint.
# ---------------------------------------------------------------------------
ENABLE_ROAD_FALLBACK = True
ROAD_DOWNSCALE        = 4      # process at 1/4 resolution (a few ms on CPU)
ROAD_TEXTURE_STD_MAX  = 7.0    # max local intensity std for 'smooth' (asphalt)
ROAD_CHROMA_MAX       = 30.0   # max Lab chroma for 'grey-ish' road
ROAD_MIN_BRIGHTNESS   = 40     # darker than this = deep shadow, not trusted
ROAD_HORIZON_FRAC     = 0.40   # ignore everything above this frame fraction
ROAD_BONNET_FRAC      = 0.93   # ignore the bonnet below this fraction
ROAD_SEED_Y_FRAC      = 0.88   # row of the seed strip (just ahead of the car)
ROAD_MIN_COVERAGE     = 0.22   # min centreline row coverage to trust the road
                               # (narrow hill roads legitimately show few rows
                               # ahead at a crest or hairpin; 0.22 ~= the
                               # ROAD_MIN_ROWS floor, keeping the gates consistent)
ROAD_MIN_ROWS         = 8      # min centreline points to trust the road
ROAD_LOOKAHEAD_FRAC   = 0.66   # centreline point used as the steering target
ROAD_SMOOTHING_ALPHA  = 0.25   # EMA on the look-ahead x (steadier steering)
ROAD_FILL_COLOR       = (0, 180, 0)   # BGR of the surface overlay
ROAD_FILL_ALPHA       = 0.35

# ---------------------------------------------------------------------------
# MODULE 1c — Learned drivable-area road following (trained CNN)
# LRASPP MobileNetV3-Large, trained on the full Indian Driving Dataset (IDD,
# ~20k images; val drivable-IoU 0.92). This is the PRIMARY guidance source on
# unmarked roads: paint (Module 1) first, this second, classical surface
# following (Module 1b) as the fallback when the model file is absent or the
# model finds no road. Runs on CPU via PyTorch (~4 fps at 768x432 on a laptop).
# Guidance is capped (LEARNED_CONF_CAP) — drivable-area, not paint-precise.
# ---------------------------------------------------------------------------
ENABLE_LEARNED_ROAD  = True
# Architecture must match the weights file:
#   "lraspp"    -> LRASPP MobileNetV3      (Phase 6c, ~6 fps @768x432, drivable_idd_full_best.pth)
#   "deeplabv3" -> DeepLabV3 MobileNetV3   (Phase 7,  ~2.6 fps @768x432, richer ASPP head + augmentation)
LEARNED_ARCH         = "lraspp"
LEARNED_MODEL_PATH   = "drivable_idd_lraspp_aug_best.pth"  # night/fog/blur/shadow fine-tune (train_local.py); was drivable_idd_full_best.pth
LEARNED_INPUT_W      = 768     # inference resolution (matches training; lower = faster, less precise)
LEARNED_INPUT_H      = 432
LEARNED_NUM_THREADS  = 0       # 0 = leave torch default; set to physical cores to cap CPU use
LEARNED_HORIZON_FRAC = 0.35    # ignore rows above this (far field / sky)
LEARNED_BONNET_FRAC  = 0.93    # ignore the bonnet below this
LEARNED_MIN_COVERAGE = 0.20    # min centreline row coverage to trust the model
LEARNED_MIN_ROWS     = 8       # min centreline points to trust the model
LEARNED_LOOKAHEAD_FRAC = 0.66  # centreline point used as the steering target
LEARNED_SMOOTHING_ALPHA = 0.25 # EMA on the look-ahead x (steadier steering)
LEARNED_CONF_CAP     = 0.80    # cap on reported confidence (est. drivable-area, not paint)
LEARNED_FILL_COLOR   = (200, 130, 0)  # BGR teal — distinct from classical green road fill
LEARNED_FILL_ALPHA   = 0.40

# ── Real-time speed (Phase 8) ────────────────────────────────────────────────
# Three stackable levers, none needs retraining:
#   BACKEND      "onnx" runs the SAME weights through ONNX Runtime — usually
#                1.5-3x faster than PyTorch on CPU, identical output. Falls back
#                to "torch" automatically if the .onnx (or onnxruntime) is absent.
#   INFER_EVERY  run the CNN once every N frames and reuse the mask in between.
#                The road barely moves frame-to-frame, so N=3 ~triples throughput
#                with negligible quality loss; the centreline still updates every
#                frame from the cached mask.
#   INPUT_W/H    (above) drop to 512x288 for ~2x, 384x216 for ~4x (less precise).
# Stacked, these turn ~3 fps into real-time on a plain laptop CPU.
LEARNED_BACKEND      = "onnx"   # "onnx" (fast) | "torch"
LEARNED_ONNX_PATH    = "drivable_idd_lraspp_768x432.onnx"  # made by export_onnx.py
LEARNED_INFER_EVERY  = 3        # run the CNN every Nth frame (1 = every frame)

# ---------------------------------------------------------------------------
# Lane smoothing
# Exponential moving average keeps the displayed lanes stable across frames.
# Lower alpha = smoother but slower to react. Higher = more responsive but jittery.
# ---------------------------------------------------------------------------
SMOOTHING_ALPHA = 0.15
LANE_KEEP_MISSES = 10   # frames a missing lane may be held from memory before it
                        # is discarded — a stale "ghost lane" held forever would
                        # blend into new scenes and corrupt fresh detections

# ---------------------------------------------------------------------------
# Steering decisions
# Offset is measured in pixels from the frame centre.
# ---------------------------------------------------------------------------
STEER_THRESHOLD_SLIGHT = 30    # pixels — "slight" correction
STEER_THRESHOLD_MODERATE = 80  # pixels — "moderate" correction
STEER_THRESHOLD_HARD = 150     # pixels — "hard" correction

# ---------------------------------------------------------------------------
# Visualisation colours  (BGR format for OpenCV)
# ---------------------------------------------------------------------------
COLOR_LEFT_LANE   = (0,   255,  0)    # Green
COLOR_RIGHT_LANE  = (0,   255,  0)    # Green
COLOR_CENTER_LINE = (0,   200, 255)   # Yellow
COLOR_ROI         = (100, 100, 100)   # Dark grey (debug mode only)
COLOR_WARNING     = (0,   0,   255)   # Red
COLOR_OK          = (0,   255,  0)    # Green
COLOR_HUD_BG      = (20,  20,  20)    # Near-black HUD background

# ---------------------------------------------------------------------------
# Debug mode — draws ROI outline and raw Hough lines when True
# ---------------------------------------------------------------------------
DEBUG_MODE = False

# ===========================================================================
# MODULE 2 — Object detection (YOLOv8n)
# ===========================================================================

# Model file — 'yolov8n.pt' is the nano version (~6 MB, fastest on CPU).
# Downloads automatically on first run. Alternatives: yolov8s.pt (more
# accurate, slower). Stick with nano on a CPU-only machine.
YOLO_MODEL = "yolov8n.pt"

YOLO_CONF_THRESHOLD = 0.35   # Minimum detection confidence
YOLO_IOU_THRESHOLD  = 0.45   # Non-max-suppression IoU threshold

# ── Speed (Phase 8) ──────────────────────────────────────────────────────────
# Measured on an i7-8650U CPU: unlike the road model, ONNX gives YOLOv8n NO real
# speedup — it's tiny and overhead-bound (preprocess + NMS dominate the forward
# pass). The lever that DOES help on CPU is input resolution, YOLO_IMGSZ:
#     640 = default/most range,  512 ~= 1.3x,  416 ~= 1.4x (misses more small /
#     distant objects — a safety trade-off, so it's left at 640 by default).
# Only the "torch" backend can change imgsz at run time; an ONNX graph is fixed
# at its export resolution.
YOLO_IMGSZ      = 640             # inference resolution (lower = faster, less range)
YOLO_BACKEND    = "torch"         # "torch" (default; flexible imgsz) | "onnx" (fixed res, ~equal on CPU)
YOLO_ONNX_MODEL = "yolov8n.onnx"  # optional; produced by export_yolo_onnx.py
# Object detection runs EVERY frame — a hazard can appear between any two frames,
# so we never frame-skip it (that's why only the road model, which barely moves
# frame-to-frame, uses LEARNED_INFER_EVERY).

# COCO class names we care about on a road. Everything else is ignored.
RELEVANT_CLASSES = {
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "train",
    "truck",
    "traffic light",
    "stop sign",
    "cat",
    "dog",
    "cow",           # relevant on Indian roads
}

# ---------------------------------------------------------------------------
# Monocular distance estimation
# ---------------------------------------------------------------------------
# Calibration reference (a typical car): ~1.5 m tall appearing ~220 px high
# at ~15 m from the camera. Adjust CALIB_* if your dashcam mounting differs.
CALIB_REAL_HEIGHT_M = 1.5
CALIB_PIXEL_HEIGHT  = 220.0
CALIB_DISTANCE_M    = 15.0

# Typical real-world heights (metres) per class, used to back out distance.
CLASS_REAL_HEIGHTS = {
    "person":     1.7,
    "bicycle":    1.1,
    "car":        1.5,
    "motorcycle": 1.3,
    "bus":        3.2,
    "truck":      3.5,
    "cow":        1.5,
    "dog":        0.6,
    "cat":        0.3,
    "default":    1.5,
}

# ---------------------------------------------------------------------------
# Path corridor — how wide the "in front of us" zone is (pixels, half-width).
# Narrows near the horizon, widens near the vehicle to mimic perspective.
# ---------------------------------------------------------------------------
PATH_CORRIDOR_MIN_PX = 60     # half-width near the horizon
PATH_CORRIDOR_MAX_PX = 300    # half-width near the vehicle

# ---------------------------------------------------------------------------
# Risk thresholds by estimated distance (metres)
# ---------------------------------------------------------------------------
RISK_DISTANCE_HIGH   = 8.0    # closer than this + in path = HIGH risk
RISK_DISTANCE_MEDIUM = 20.0   # closer than this = MEDIUM risk

# ---------------------------------------------------------------------------
# Module toggles
# ---------------------------------------------------------------------------
ENABLE_LANE_DETECTION   = True
ENABLE_OBJECT_DETECTION = True

# ===========================================================================
# MODULE 3 — Decision engine (sensor fusion + arbitration)
# ===========================================================================
# The decision engine reads only the *results* of Modules 1 and 2 (never
# pixels) and fuses them into one arbitrated driving action per frame. All of
# its behaviour is governed by the constants below.
#
# Mental model of the five longitudinal levels:
#   0 PROCEED         maintain speed, follow lane freely
#   1 CAUTION         ease off throttle, cover the brake
#   2 SLOW            active moderate braking
#   3 BRAKE           firm proportional braking
#   4 EMERGENCY_STOP  maximum straight-line braking (latched)
# ---------------------------------------------------------------------------

# --- Longitudinal distance / time-to-collision (TTC) thresholds -------------
DIST_EMERGENCY_M    = 5.0    # in-path object closer than this  -> EMERGENCY
TTC_EMERGENCY_S     = 1.2    # in-path TTC (while closing) below this -> EMERGENCY
TTC_BRAKE_S         = 2.5    # in-path TTC below this -> BRAKE
TTC_CAUTION_S       = 4.0    # in-path TTC below this -> CAUTION (gentle closing)
MIN_CLOSING_MPS     = 0.3    # below this closing speed, TTC is undefined (receding)
STOPSIGN_DISTANCE_M = 25.0   # in-path stop sign / light within this range -> SLOW

# --- Per-object kinematics smoothing ----------------------------------------
# Used by the Module 4 tracker to smooth each track's noisy monocular distance
# and derive its closing speed + TTC. (Brief detection dropouts are handled by
# the tracker's coast/max-age lifecycle, see Module 4 below.)
DIST_EMA_ALPHA       = 0.4   # EMA weight on new distance sample (higher = snappier)
CLOSING_EMA_ALPHA    = 0.5   # EMA weight on derived closing speed
MAX_PLAUSIBLE_JUMP_M = 8.0   # frame-to-frame jump above this = monocular spike
MAX_DIST_STEP_M      = 3.0   # max distance step applied when a spike is rejected
VCLOSE_CLAMP_MPS     = 40.0  # clamp on smoothed closing-speed magnitude
DT_CLAMP_MIN_S       = 0.02  # clamp on measured per-frame dt (survives fps jitter)
DT_CLAMP_MAX_S       = 0.5

# --- Temporal ratchet (debounce so one bad frame can't flip the action) -----
# (N, M): the rule for a level must fire in N of the last M frames to escalate
# to it. Escalation is fast (and may jump multiple levels at once); release is
# slow (one level at a time, gated by HOLD_FRAMES).
ESC_EMERGENCY = (2, 3)   # a genuine emergency latches in 2 frames
ESC_BRAKE     = (3, 5)   # a lone spurious HIGH box (fails 3-of-5) never reaches BRAKE
ESC_SLOW      = (2, 3)
ESC_CAUTION   = (2, 3)
EVIDENCE_WINDOW        = 5   # length of the raw-level history (>= max M above)
HOLD_FRAMES            = 8   # consecutive calmer frames before stepping DOWN one level
HOLD_FRAMES_DEGRADED   = 12  # slower release while inputs are degraded
EMERGENCY_LATCH_FRAMES = 15  # minimum dwell in EMERGENCY_STOP before any downgrade

# --- Degraded-mode trust gates ----------------------------------------------
MIN_FPS                = 2.0   # below this effective fps, temporal reasoning is degraded.
                               # NOTE: YOLOv8n on a CPU runs ~3-5 fps — that is the NORMAL
                               # operating point here, so this floor is deliberately low and
                               # only trips on a genuine stall, not on ordinary CPU cadence.
DET_CONF_MIN           = 0.50  # nearest-in-path confidence below this -> degraded.
                               # MUST stay above YOLO_CONF_THRESHOLD (0.35), else YOLO has
                               # already filtered everything below it and this never trips.
DEGRADED_DIST_MARGIN_M = 5.0   # widen the MEDIUM-in-path SLOW band when degraded
DEGRADED_TTC_MARGIN_S  = 1.0   # react earlier (add to every TTC threshold) when degraded

# --- Class-aware reaction ---------------------------------------------------
# Vulnerable road users earn extra reaction margin; advisory signs are never
# braked on hard (Module 2 does not report a light's colour, so we only ease).
VULNERABLE_CLASSES      = {"person", "bicycle", "motorcycle", "cow", "dog", "cat"}
VULNERABLE_TTC_MARGIN_S = 0.8
ADVISORY_CLASSES        = {"traffic light", "stop sign"}

# --- Brake / throttle command scalars (0.0 - 1.0) ---------------------------
BRAKE_EASE = 0.15   # CAUTION
BRAKE_SLOW = 0.30   # SLOW
BRAKE_MIN  = 0.35   # BRAKE lower bound (also used for a static HIGH object)
BRAKE_MAX  = 0.85   # BRAKE upper bound (EMERGENCY uses 1.0)

# --- Decision HUD colours (BGR) ---------------------------------------------
COLOR_PROCEED   = (0, 200, 0)     # green
COLOR_CAUTION   = (0, 200, 255)   # amber
COLOR_SLOW      = (0, 140, 255)   # deep amber
COLOR_BRAKE     = (0, 90, 255)    # orange
COLOR_EMERGENCY = (0, 0, 255)     # red
COLOR_DEGRADED  = (0, 215, 255)   # yellow

# --- Module toggle ----------------------------------------------------------
ENABLE_DECISION_ENGINE = True

# ===========================================================================
# MODULE 4 — Multi-object tracker (stable IDs + per-object kinematics)
# ===========================================================================
# Gives each detected object a stable identity across frames so the decision
# engine can follow specific objects (steadier decisions in dense traffic)
# instead of re-deriving everything from a flickering "nearest" box.

TRACK_IOU_MIN  = 0.30   # min IoU to associate a detection with an existing track
TRACK_MAX_AGE  = 5      # frames a track may coast (unmatched) before it is dropped
TRACK_MIN_HITS = 3      # detections before a track is 'confirmed' — rejects 1-frame ghosts
TRACK_BBOX_EMA = 0.5    # EMA weight smoothing each track's bounding box

COLOR_TRACK_ID = (255, 255, 255)   # track-ID tag colour (BGR)

# --- Module toggle ----------------------------------------------------------
ENABLE_TRACKING = True

# ===========================================================================
# PHASE 9 — Advisory simulation layers (Modules 5-9)
# ===========================================================================
# Software demonstrations of the perception -> fusion -> prediction -> planning
# -> control chain, plus driver monitoring. ALL SIMULATION / ADVISORY ONLY —
# none of this is ever connected to a real vehicle's sensors or actuators.
# The "radar" is synthesised, the "control" drives a simulated ego only.
# See ROADMAP.md and the README safety notice.

# --- Layer 3: bird's-eye (ego-frame) projection, the shared 3D-ish space -----
ENABLE_BEV        = True
BEV_RANGE_M       = 60.0    # forward distance shown in the bird's-eye panel
BEV_HALF_WIDTH_M  = 12.0    # lateral half-width shown (+/-)
BEV_PANEL_W       = 200     # panel size in px
BEV_PANEL_H       = 260
COLOR_BEV_BG      = (25, 25, 25)
COLOR_BEV_EGO     = (0, 220, 0)
COLOR_BEV_CAM     = (255, 180, 0)   # camera depth estimate (BGR)
COLOR_BEV_RADAR   = (0, 180, 255)   # simulated-radar estimate
COLOR_BEV_FUSED   = (255, 255, 255) # fused estimate

# --- Layer 1: simulated sensor fusion ----------------------------------------
ENABLE_FUSION          = True
RADAR_RANGE_NOISE_M    = 0.8   # sim-radar range noise (1 sigma)
RADAR_VEL_NOISE_MPS    = 0.4   # sim-radar range-rate noise (1 sigma)
FUSION_CAM_RANGE_VAR   = 9.0   # camera depth is noisy (variance, m^2)
FUSION_RADAR_RANGE_VAR = 0.6   # radar range is precise (variance, m^2)

# --- Layer 4: prediction & planning ------------------------------------------
ENABLE_PLANNING     = True
PRED_HORIZON_S      = 3.0     # how far ahead objects are predicted
PRED_STEP_S         = 0.5     # prediction sampling step
PLAN_CRUISE_MPS     = 13.9    # nominal cruise target (~50 km/h)
PLAN_TIME_GAP_S     = 2.0     # desired following time gap
PLAN_MIN_GAP_M      = 6.0     # minimum standoff distance
COLOR_PLAN_PATH     = (0, 255, 180)   # advisory path ribbon
COLOR_PRED_ARROW    = (180, 120, 255) # predicted object motion (BEV)

# --- Layer 5: simulated control — NEVER wired to a vehicle -------------------
ENABLE_CONTROL_SIM  = True
CTRL_WHEELBASE_M    = 2.7
CTRL_LOOKAHEAD_M    = 8.0     # pure-pursuit look-ahead
CTRL_MAX_STEER_DEG  = 35.0
CTRL_KP = 0.6                 # longitudinal PID
CTRL_KI = 0.05
CTRL_KD = 0.0
CTRL_SIM_INIT_MPS   = 13.9    # simulated ego speed at start
CTRL_MAX_ACCEL_MPS2 = 2.5
CTRL_MAX_BRAKE_MPS2 = 6.0

# --- Layer 7: driver monitoring — needs a driver-facing camera --------------
ENABLE_DRIVER_MON      = True   # only runs when --driver-cam INDEX is provided
DMS_EYES_CLOSED_FRAMES = 12     # consecutive no-eye frames -> DROWSY
DMS_LOOKAWAY_RATIO     = 0.22   # face-centre offset (frac of width) -> DISTRACTED
