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
# Lane smoothing
# Exponential moving average keeps the displayed lanes stable across frames.
# Lower alpha = smoother but slower to react. Higher = more responsive but jittery.
# ---------------------------------------------------------------------------
SMOOTHING_ALPHA = 0.15

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
