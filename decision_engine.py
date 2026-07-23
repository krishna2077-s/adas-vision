"""
decision_engine.py — Module 3: the decision engine ("the brain").

Phases 1 and 2 are the eyes. This module is the brain. It never touches pixels
for perception — it reasons over the *structured results* the two perception
modules already produce and fuses them into ONE arbitrated driving action per
frame, plus a single human-readable reason a driver or a log would understand.

How it works (end to end, once per frame):

    1. Trust check        — set valid / degraded flags from the two results.
    2. Threat tracker     — EMA-smooth the noisy monocular distance to the
                            nearest in-path object, reject spikes, derive a
                            smoothed closing speed and a real-seconds TTC, and
                            hold the hazard through brief detection dropouts.
    3. Policy table       — an ordered, first-match-wins list of rules R1..R7
                            (collision rules on top) yields this frame's RAW
                            longitudinal level 0-4.
    4. Temporal ratchet   — convert RAW -> COMMITTED. Escalation is fast
                            (N-of-M, may jump levels); release is slow (one
                            level per HOLD_FRAMES) with an emergency latch. So
                            no single noisy frame can flip the action.
    5. Lateral arbitration— derive a steering action from the lane module and
                            SAFETY-CLAMP it: lanes can only ever *reduce*
                            lateral authority, never raise throttle or lower the
                            brake, and never steer toward a hazard.
    6. Output             — one DrivingDecision with committed longitudinal +
                            lateral actions, throttle/brake scalars, the winning
                            rule id, telemetry, and the reason string.

Safety is by construction: collision rules sit at the top of the table and the
lane path can never weaken the longitudinal decision.

The engine adds negligible CPU (a handful of scalar ops, one length-5 deque, a
few integer counters) and no new dependency beyond cv2/config already in use.
"""

import logging
from collections import deque
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import cv2

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Longitudinal levels
# ---------------------------------------------------------------------------

PROCEED, CAUTION, SLOW, BRAKE, EMERGENCY_STOP = 0, 1, 2, 3, 4
LEVEL_NAMES = ["PROCEED", "CAUTION", "SLOW", "BRAKE", "EMERGENCY_STOP"]

_LEVEL_COLORS = {
    PROCEED:        cfg.COLOR_PROCEED,
    CAUTION:        cfg.COLOR_CAUTION,
    SLOW:           cfg.COLOR_SLOW,
    BRAKE:          cfg.COLOR_BRAKE,
    EMERGENCY_STOP: cfg.COLOR_EMERGENCY,
}


# ---------------------------------------------------------------------------
# Output data class
# ---------------------------------------------------------------------------

@dataclass
class DrivingDecision:
    """The single fused driving decision for one frame."""
    valid:              bool  = True    # False -> holding the previous decision
    degraded:           bool  = False   # inputs usable but reduced-trust
    longitudinal:       str   = "PROCEED"
    longitudinal_level: int   = PROCEED
    lateral:            str   = "KEEP_LANE"   # KEEP_LANE / CORRECT_LEFT / CORRECT_RIGHT / HOLD
    lateral_magnitude:  str   = "NONE"        # NONE / SLIGHT / MODERATE / HARD
    throttle:           float = 1.0
    brake:              float = 0.0
    reason:             str   = ""
    rule_id:            str   = "R7"
    hazard_label:       Optional[str]   = None
    smoothed_distance_m: Optional[float] = None
    closing_speed_mps:  Optional[float] = None
    ttc_s:              Optional[float] = None
    hazard_side:        Optional[str]   = None   # 'LEFT' / 'RIGHT' / None
    lane_confidence:    float = 0.0
    frame_index:        int   = 0
    fps:                float = 0.0


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

class DecisionEngine:
    """
    Stateful fusion + arbitration engine.

    Usage::

        engine   = DecisionEngine(frame_width=1280, frame_height=720)
        decision = engine.process(lane_result, obj_result)   # per frame
        annotated = engine.draw_hud(annotated, decision)
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height

        # ── Committed-level ratchet state ─────────────────────────────
        self.committed_level = PROCEED
        self.raw_history: "deque[int]" = deque(maxlen=cfg.EVIDENCE_WINDOW)
        self.down_counter = 0
        self.emergency_dwell = 0

        # ── Threat tracker state ──────────────────────────────────────
        self._tracked = False
        self._smoothed_distance: Optional[float] = None
        self._closing_speed = 0.0
        self._ttc: Optional[float] = None
        self._lost_grace = 0
        self._hazard = None                 # last known nearest-in-path Detection

        # ── Timing / bookkeeping ──────────────────────────────────────
        self._prev_tick = cv2.getTickCount()
        self.frame_index = 0
        self._last_decision: Optional[DrivingDecision] = None

        logger.info(f"DecisionEngine initialised ({frame_width}x{frame_height})")

    def reset(self) -> None:
        """Clear all temporal state — call when a video loops back to the start
        so the replay begins from a clean PROCEED baseline (not mid-brake)."""
        self.committed_level = PROCEED
        self.raw_history.clear()
        self.down_counter = 0
        self.emergency_dwell = 0
        self._tracked = False
        self._smoothed_distance = None
        self._closing_speed = 0.0
        self._ttc = None
        self._lost_grace = 0
        self._hazard = None
        self._last_decision = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, lane_result, obj_result) -> DrivingDecision:
        """
        Fuse one frame of lane + object results into a driving decision.

        Either argument may be None (its module was disabled or unavailable);
        the engine degrades gracefully. If BOTH are None it holds the previous
        decision rather than inventing one.
        """
        self.frame_index += 1
        raw_dt, dt = self._measure_dt()

        # ── Validity: nothing to reason about ─────────────────────────
        if lane_result is None and obj_result is None:
            return self._hold_or_default()

        # ── Trust / degraded flags ────────────────────────────────────
        # fps from the UNCLAMPED dt so a real stall crosses the MIN_FPS gate.
        lane_conf = lane_result.confidence if lane_result is not None else 0.0
        nearest = self._nearest_actionable(obj_result)
        fps = 1.0 / raw_dt if raw_dt > 1e-6 else 999.0

        degraded, degraded_cause = self._assess_trust(
            lane_result, obj_result, nearest, fps
        )

        # ── Threat tracker (updates smoothed distance / closing / TTC) ─
        hazard = self._update_threat(nearest, dt)
        hazard_side = self._hazard_side(hazard)

        # ── Policy table -> raw longitudinal level ────────────────────
        raw_level, rule_id, reason_core = self._evaluate_rules(
            hazard, degraded, obj_result
        )

        # ── Temporal ratchet: raw -> committed ────────────────────────
        committed = self._ratchet(raw_level, degraded)

        # ── Degraded floor: never claim PROCEED while blind + hazard ──
        highest_risk = obj_result.highest_risk if obj_result is not None else "LOW"
        floored = False
        if degraded and committed < CAUTION and (
            hazard is not None or highest_risk in ("MEDIUM", "HIGH")
        ):
            committed = CAUTION
            floored = True
        self.committed_level = committed
        self.emergency_dwell = self.emergency_dwell + 1 if committed == EMERGENCY_STOP else 0

        # ── Lateral action, safety-arbitrated ─────────────────────────
        lateral, lateral_mag, lateral_note = self._lateral(
            lane_result, committed, hazard_side
        )

        # ── Command scalars ───────────────────────────────────────────
        throttle, brake = self._scalars(committed)

        # ── Assemble the reason string (honest about WHY committed level) ──
        core_text = reason_core.split("] ", 1)[-1]
        if floored:
            rid = "FLOOR"
            reason = "[FLOOR] CAUTION: easing off -- inputs untrusted with a hazard nearby"
        elif committed == raw_level:
            rid = rule_id
            reason = reason_core
        elif committed > raw_level:
            # Ratchet is holding a calmer-than-committed read (de-escalating slowly).
            rid = rule_id
            reason = f"[{rule_id}] {LEVEL_NAMES[committed]}: holding -- live read {core_text}"
        else:
            # Ratchet is still confirming an escalation (raw is higher than committed).
            rid = rule_id
            reason = f"[{rule_id}] {LEVEL_NAMES[committed]}: confirming {LEVEL_NAMES[raw_level]} -- {core_text}"

        if lateral_note:
            reason = f"{reason}; {lateral_note}"
        if degraded:
            reason = f"[DEGRADED] {reason}"
            if degraded_cause and degraded_cause not in reason:
                reason = f"{reason} ({degraded_cause})"

        decision = DrivingDecision(
            valid=True,
            degraded=degraded,
            longitudinal=LEVEL_NAMES[committed],
            longitudinal_level=committed,
            lateral=lateral,
            lateral_magnitude=lateral_mag,
            throttle=throttle,
            brake=brake,
            reason=reason,
            rule_id=rid,
            hazard_label=hazard.label if hazard is not None else None,
            smoothed_distance_m=(round(self._smoothed_distance, 1)
                                 if self._smoothed_distance is not None else None),
            closing_speed_mps=round(self._closing_speed, 1),
            ttc_s=round(self._ttc, 1) if self._ttc is not None else None,
            hazard_side=hazard_side,
            lane_confidence=lane_conf,
            frame_index=self.frame_index,
            fps=fps,
        )
        self._last_decision = decision
        return decision

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _measure_dt(self) -> Tuple[float, float]:
        """
        Returns (raw_dt, clamped_dt) in seconds since the previous call.

        raw_dt drives the fps trust gate (so a genuine stall can be *seen* —
        the clamp would otherwise floor fps at exactly MIN_FPS and hide it).
        clamped_dt drives the EMA / closing-speed integration (kept bounded so
        fps jitter can't blow up the derivative).
        """
        tick = cv2.getTickCount()
        raw = (tick - self._prev_tick) / cv2.getTickFrequency()
        self._prev_tick = tick
        clamped = max(cfg.DT_CLAMP_MIN_S, min(cfg.DT_CLAMP_MAX_S, raw))
        return raw, clamped

    # ------------------------------------------------------------------
    # Trust / degraded assessment
    # ------------------------------------------------------------------

    def _assess_trust(self, lane_result, obj_result, nearest, fps) -> Tuple[bool, str]:
        """Returns (degraded, first_cause_string)."""
        if lane_result is None:
            return True, "object-only mode"
        if lane_result.confidence == 0.0:
            return True, "no lane lock -- corridor assumed frame-centre"
        if obj_result is None:
            return True, "lane-only mode"
        if fps < cfg.MIN_FPS:
            return True, f"low fps ({fps:.1f})"
        if nearest is not None and nearest.confidence < cfg.DET_CONF_MIN:
            return True, "low-confidence detection"
        return False, ""

    # ------------------------------------------------------------------
    # Hazard selection
    # ------------------------------------------------------------------

    def _nearest_actionable(self, obj_result):
        """
        The nearest in-path object the COLLISION rules should act on — i.e. the
        closest non-advisory in-path detection.

        We deliberately do NOT reuse obj_result.nearest_in_path: that is the
        class-agnostic closest in-path box, so a traffic light / stop sign
        estimated nearer than a lead car would become the tracked hazard and
        (being advisory) would suppress braking for the vehicle behind it.
        Advisory signs are handled separately by rule R4, which scans every
        detection, so excluding them here loses nothing.
        """
        if obj_result is None:
            return None
        candidates = [
            d for d in obj_result.detections
            if d.in_path and d.label not in cfg.ADVISORY_CLASSES
        ]
        return min(candidates, key=lambda d: d.distance_m) if candidates else None

    # ------------------------------------------------------------------
    # Threat tracker  (smooths distance, derives closing speed + TTC)
    # ------------------------------------------------------------------

    def _update_threat(self, nearest, dt):
        """
        Update the single-object threat track and return the 'effective
        hazard' Detection to reason about this frame — the live nearest object,
        a retained one during a brief dropout, or None.
        """
        if nearest is not None:
            raw_d = nearest.distance_m

            if not self._tracked:
                # First sighting (or first frame after a reset): seed the track.
                self._smoothed_distance = raw_d
                self._closing_speed = 0.0
                self._ttc = None
                self._tracked = True
            else:
                prev = self._smoothed_distance
                jump = abs(raw_d - prev)
                if jump <= cfg.MAX_PLAUSIBLE_JUMP_M:
                    # Plausible movement: EMA the distance, derive closing speed.
                    self._smoothed_distance = (
                        cfg.DIST_EMA_ALPHA * raw_d
                        + (1.0 - cfg.DIST_EMA_ALPHA) * prev
                    )
                    inst_v = (prev - self._smoothed_distance) / dt   # + = approaching
                    self._closing_speed = (
                        cfg.CLOSING_EMA_ALPHA * inst_v
                        + (1.0 - cfg.CLOSING_EMA_ALPHA) * self._closing_speed
                    )
                    self._closing_speed = max(
                        -cfg.VCLOSE_CLAMP_MPS, min(cfg.VCLOSE_CLAMP_MPS, self._closing_speed)
                    )
                else:
                    # Spike or object-identity switch: step toward the reading by
                    # a bounded amount and drop the closing estimate to zero. The
                    # jump usually means nearest_in_path swapped to a different
                    # physical object, so any closing speed derived across the two
                    # would be fiction — and (with TTC below) it must not fabricate
                    # a phantom time-to-collision. Closing rebuilds once distance
                    # settles back into the plausible band.
                    step = min(cfg.MAX_DIST_STEP_M, jump)
                    self._smoothed_distance = prev + step if raw_d > prev else prev - step
                    self._closing_speed = 0.0

            self._ttc = (
                self._smoothed_distance / self._closing_speed
                if self._closing_speed > cfg.MIN_CLOSING_MPS else None
            )
            self._lost_grace = 0
            self._hazard = nearest
            return nearest

        # nearest is None ---------------------------------------------------
        if self._tracked and self._lost_grace < cfg.OBJECT_LOST_GRACE_FRAMES:
            # Brief dropout: keep the track *state* alive (decay distance "away",
            # so a reappearing object doesn't re-seed and spike the closing speed)
            # but return NO hazard to the rule table. This matters: grace must not
            # manufacture escalation evidence — one spurious detection followed by
            # grace frames must never satisfy the N-of-M escalation debounce. An
            # already-committed brake is instead held by the ratchet's down-counter
            # (HOLD_FRAMES), which is a strictly stronger hold than this window.
            self._lost_grace += 1
            self._smoothed_distance += cfg.CLEAR_RATE_MPS * dt
            self._closing_speed = 0.0
            self._ttc = None
            return None

        # Fully lost: reset the track.
        self._tracked = False
        self._smoothed_distance = None
        self._closing_speed = 0.0
        self._ttc = None
        self._hazard = None
        return None

    def _hazard_side(self, hazard) -> Optional[str]:
        """Which side of frame-centre the hazard sits on ('LEFT'/'RIGHT'/None)."""
        if hazard is None:
            return None
        hx = hazard.center[0]
        cx = self.w // 2
        if abs(hx - cx) < cfg.STEER_THRESHOLD_SLIGHT:
            return None                      # dead ahead — no side bias
        return "LEFT" if hx < cx else "RIGHT"

    # ------------------------------------------------------------------
    # Policy table  (ordered, first-match-wins -> raw longitudinal level)
    # ------------------------------------------------------------------

    def _evaluate_rules(self, hazard, degraded, obj_result) -> Tuple[int, str, str]:
        d = self._smoothed_distance
        ttc = self._ttc
        closing = self._closing_speed > cfg.MIN_CLOSING_MPS

        detections = obj_result.detections if obj_result is not None else []
        highest_risk = obj_result.highest_risk if obj_result is not None else "LOW"

        # ── Rules needing the nearest actionable hazard (R1..R3) ──────
        # `hazard` is the nearest NON-advisory in-path object (see
        # _nearest_actionable), so collision rules always see the real threat
        # even when a traffic light / sign sits closer in the same corridor.
        if hazard is not None:
            label = hazard.label

            ttc_margin = cfg.DEGRADED_TTC_MARGIN_S if degraded else 0.0
            if label in cfg.VULNERABLE_CLASSES:
                ttc_margin += cfg.VULNERABLE_TTC_MARGIN_S
            ttc_emerg  = cfg.TTC_EMERGENCY_S + ttc_margin
            ttc_brake  = cfg.TTC_BRAKE_S     + ttc_margin

            # R1 — EMERGENCY_STOP
            ttc_emerg_hit = ttc is not None and ttc <= ttc_emerg and closing
            if d <= cfg.DIST_EMERGENCY_M or ttc_emerg_hit:
                if ttc_emerg_hit:
                    r = f"[R1] EMERGENCY_STOP: {label} {d:.1f}m in-path, TTC {ttc:.1f}s closing"
                else:
                    r = f"[R1] EMERGENCY_STOP: {label} {d:.1f}m in-path"
                return EMERGENCY_STOP, "R1", r

            # R2 — BRAKE (closing fast)
            if ttc is not None and ttc <= ttc_brake and closing:
                vuln = " (vulnerable, early brake)" if label in cfg.VULNERABLE_CLASSES else ""
                return BRAKE, "R2", f"[R2] BRAKE: {label} closing, {d:.1f}m TTC {ttc:.1f}s{vuln}"

            # R3 — BRAKE (static HIGH-risk object dead ahead)
            if hazard.risk == "HIGH" and d <= cfg.RISK_DISTANCE_HIGH:
                return BRAKE, "R3", f"[R3] BRAKE: {label} {d:.1f}m in-path (HIGH)"

        # ── R4 — SLOW for an in-path traffic control (advisory ease) ──
        # We only EASE, never stop hard: Module 2 reports no light colour.
        advisory_hit = None
        for det in detections:
            if (det.in_path and det.label in cfg.ADVISORY_CLASSES
                    and det.distance_m <= cfg.STOPSIGN_DISTANCE_M):
                if advisory_hit is None or det.distance_m < advisory_hit.distance_m:
                    advisory_hit = det
        if advisory_hit is not None:
            return SLOW, "R4", f"[R4] SLOW: {advisory_hit.label} {advisory_hit.distance_m:.0f}m ahead"

        # ── R5 — SLOW for a MEDIUM-risk in-path object ────────────────
        if hazard is not None:
            margin = cfg.DEGRADED_DIST_MARGIN_M if degraded else 0.0
            if hazard.risk == "MEDIUM" and d <= cfg.RISK_DISTANCE_MEDIUM + margin:
                return SLOW, "R5", f"[R5] SLOW: {hazard.label} {d:.1f}m in-path (MEDIUM)"

        # ── R6 — CAUTION (gentle in-path closing, or any nearby risk) ─
        # Reaching here means no in-path rule fired above, so the tracked
        # hazard (if any) is benign. Still ease off if it is gently closing, or
        # if ANY object anywhere carries MEDIUM/HIGH risk (e.g. a pedestrian off
        # to the side — which must not be ignored just because a far car is ahead).
        if hazard is not None and closing and ttc is not None:
            ttc_margin = cfg.DEGRADED_TTC_MARGIN_S if degraded else 0.0
            if hazard.label in cfg.VULNERABLE_CLASSES:
                ttc_margin += cfg.VULNERABLE_TTC_MARGIN_S
            if (cfg.TTC_BRAKE_S + ttc_margin) < ttc <= (cfg.TTC_CAUTION_S + ttc_margin):
                return CAUTION, "R6", f"[R6] CAUTION: {hazard.label} closing gently, TTC {ttc:.1f}s"
        if highest_risk in ("MEDIUM", "HIGH"):
            return CAUTION, "R6", f"[R6] CAUTION: {highest_risk.lower()}-risk object nearby"

        # ── R7 — PROCEED (default) ────────────────────────────────────
        return PROCEED, "R7", "[R7] PROCEED: path clear"

    # ------------------------------------------------------------------
    # Temporal ratchet  (raw level -> committed level)
    # ------------------------------------------------------------------

    def _esc_requirement(self, level: int) -> Tuple[int, int]:
        return {
            EMERGENCY_STOP: cfg.ESC_EMERGENCY,
            BRAKE:          cfg.ESC_BRAKE,
            SLOW:           cfg.ESC_SLOW,
            CAUTION:        cfg.ESC_CAUTION,
        }.get(level, (1, 1))

    def _ratchet(self, raw_level: int, degraded: bool) -> int:
        """
        Fast N-of-M escalation (may jump several levels at once); slow,
        one-level-at-a-time de-escalation gated by HOLD_FRAMES with an
        EMERGENCY latch. Guarantees no single noisy frame flips the action.
        """
        self.raw_history.append(raw_level)
        committed = self.committed_level

        if raw_level > committed:
            # Any read hotter than the committed level breaks the calm streak,
            # even if it is not yet confirmed enough to escalate — otherwise a
            # lone spike between calm frames would let a brake release on
            # non-consecutive calm frames.
            self.down_counter = 0
            # Escalate to the highest level in (committed, raw] confirmed N-of-M.
            target = committed
            for lvl in range(committed + 1, raw_level + 1):
                n, m = self._esc_requirement(lvl)
                recent = list(self.raw_history)[-m:]
                if sum(1 for r in recent if r >= lvl) >= n:
                    target = lvl
            if target > committed:
                committed = target

        elif raw_level < committed:
            self.down_counter += 1
            hold = cfg.HOLD_FRAMES_DEGRADED if degraded else cfg.HOLD_FRAMES
            latched = (committed == EMERGENCY_STOP
                       and self.emergency_dwell < cfg.EMERGENCY_LATCH_FRAMES)
            if self.down_counter >= hold and not latched:
                committed -= 1
                self.down_counter = 0

        else:  # raw_level == committed
            self.down_counter = 0

        return committed

    # ------------------------------------------------------------------
    # Lateral action  (from lanes, then safety-clamped)
    # ------------------------------------------------------------------

    def _lateral(self, lane_result, committed: int, hazard_side) -> Tuple[str, str, Optional[str]]:
        """
        Derive the lateral action from the lane offset, then arbitrate it
        against the longitudinal decision. Lanes can only ever *reduce* lateral
        authority — never raise throttle or lower the brake.
        """
        # No trustworthy lane lock -> hold the wheel.
        if lane_result is None or lane_result.confidence == 0.0:
            return "HOLD", "NONE", None

        # Direction + magnitude straight from the offset (same thresholds as
        # Module 1's steering word, so they always agree).
        offset = lane_result.offset_px
        a = abs(offset)
        if a < cfg.STEER_THRESHOLD_SLIGHT:
            action, mag = "KEEP_LANE", "NONE"
        else:
            action = "CORRECT_LEFT" if offset > 0 else "CORRECT_RIGHT"
            if a < cfg.STEER_THRESHOLD_MODERATE:
                mag = "SLIGHT"
            elif a < cfg.STEER_THRESHOLD_HARD:
                mag = "MODERATE"
            else:
                mag = "HARD"

        note = None

        # Emergency braking: go straight, hold the wheel.
        if committed == EMERGENCY_STOP:
            return "HOLD", "NONE", "straight-line braking"

        # Braking: never steer *toward* the hazard (evasion away is allowed).
        if committed >= BRAKE and action != "KEEP_LANE":
            toward = ((action == "CORRECT_LEFT" and hazard_side == "LEFT")
                      or (action == "CORRECT_RIGHT" and hazard_side == "RIGHT"))
            if toward:
                return "HOLD", "NONE", "lateral inhibited: correction toward hazard"

        # Slowing or harder: cap the aggressiveness of any correction.
        if committed >= SLOW and mag in ("MODERATE", "HARD"):
            mag = "SLIGHT"

        return action, mag, note

    # ------------------------------------------------------------------
    # Command scalars
    # ------------------------------------------------------------------

    def _scalars(self, level: int) -> Tuple[float, float]:
        """Map a committed level to (throttle, brake) in 0.0-1.0."""
        if level == PROCEED:
            return 1.0, 0.0
        if level == CAUTION:
            return 0.0, cfg.BRAKE_EASE
        if level == SLOW:
            return 0.0, cfg.BRAKE_SLOW
        if level == BRAKE:
            if self._ttc is not None and self._ttc > 0:
                b = 1.0 - self._ttc / cfg.TTC_BRAKE_S
                b = max(cfg.BRAKE_MIN, min(cfg.BRAKE_MAX, b))
            else:
                b = cfg.BRAKE_MIN
            return 0.0, b
        return 0.0, 1.0   # EMERGENCY_STOP

    # ------------------------------------------------------------------
    # Hold / default when inputs are unusable
    # ------------------------------------------------------------------

    def _hold_or_default(self) -> DrivingDecision:
        if self._last_decision is not None:
            prev = self._last_decision
            held = replace(
                prev,
                valid=False,
                frame_index=self.frame_index,
                reason=f"[R0] holding last decision -- inputs invalid (was {prev.longitudinal})",
            )
            self._last_decision = held
            return held
        default = DrivingDecision(
            valid=False,
            reason="[R0] no inputs yet -- default PROCEED",
            frame_index=self.frame_index,
        )
        self._last_decision = default
        return default

    # ------------------------------------------------------------------
    # HUD
    # ------------------------------------------------------------------

    def draw_hud(self, frame, decision: DrivingDecision):
        """Draw the decision panel (top-right) and the reason strip (bottom)."""
        self._draw_panel(frame, decision)
        self._draw_reason_strip(frame, decision)
        return frame

    def _draw_panel(self, frame, d: DrivingDecision) -> None:
        pw, ph = 320, 150
        x0 = max(10, self.w - pw - 10)
        y0 = 10

        # On narrow frames the right-anchored panel would sit on top of the
        # centred Forward Collision Warning banner (top ~6..50) and, being an
        # opaque fill, hide it. Drop the panel below the banner band in that case
        # — never let the decision panel occlude the collision warning.
        banner_left = self.w // 2 - 240
        if x0 < banner_left + 480:
            y0 = 58

        cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), cfg.COLOR_HUD_BG, -1)
        cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), (80, 80, 80), 1)

        # ── State badge (blinks while EMERGENCY) ──────────────────────
        color = _LEVEL_COLORS[d.longitudinal_level]
        show = not (d.longitudinal_level == EMERGENCY_STOP and d.frame_index % 2 == 0)
        badge = d.longitudinal if show else ""
        cv2.putText(frame, badge, (x0 + 12, y0 + 34),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, color, 2, cv2.LINE_AA)

        # ── Longitudinal + brake bar ──────────────────────────────────
        cv2.putText(frame, f"LONG  {d.longitudinal}", (x0 + 12, y0 + 62),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        self._draw_brake_bar(frame, x0 + 150, y0 + 52, d.brake, color)

        # ── Lateral ───────────────────────────────────────────────────
        arrow = {"CORRECT_LEFT": "<", "CORRECT_RIGHT": ">"}.get(d.lateral, "|")
        lat_color = (120, 120, 120) if d.lateral == "HOLD" else (200, 200, 200)
        cv2.putText(frame, f"LAT   {arrow} {d.lateral} {d.lateral_magnitude}",
                    (x0 + 12, y0 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.5, lat_color, 1, cv2.LINE_AA)

        # ── Nearest-in-path summary ───────────────────────────────────
        if d.hazard_label:
            dist = f"{d.smoothed_distance_m:.0f}m" if d.smoothed_distance_m is not None else "--"
            ttc = f"{d.ttc_s:.1f}s" if d.ttc_s is not None else "--"
            trend = "^" if (d.closing_speed_mps or 0) > cfg.MIN_CLOSING_MPS else "v"
            ahead = f"AHEAD {d.hazard_label} {dist} TTC {ttc} {trend}"
        else:
            ahead = "AHEAD  clear"
        cv2.putText(frame, ahead, (x0 + 12, y0 + 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Telemetry line ────────────────────────────────────────────
        v = d.closing_speed_mps if d.closing_speed_mps is not None else 0.0
        cv2.putText(frame, f"v {v:+.1f} m/s   fps {d.fps:.0f}", (x0 + 12, y0 + 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

        # ── DEGRADED / INVALID chips ──────────────────────────────────
        if d.degraded and d.frame_index % 2 == 0:
            cv2.putText(frame, "DEGRADED", (x0 + pw - 100, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, cfg.COLOR_DEGRADED, 1, cv2.LINE_AA)
        if not d.valid:
            cv2.putText(frame, "!", (x0 + pw - 22, y0 + 34),
                        cv2.FONT_HERSHEY_DUPLEX, 0.8, cfg.COLOR_EMERGENCY, 2, cv2.LINE_AA)

    def _draw_brake_bar(self, frame, x, y, brake: float, color) -> None:
        bw, bh = 150, 12
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (50, 50, 50), -1)
        fill = int(max(0.0, min(1.0, brake)) * bw)
        if fill > 0:
            cv2.rectangle(frame, (x, y), (x + fill, y + bh), color, -1)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (150, 150, 150), 1)

    def _draw_reason_strip(self, frame, d: DrivingDecision) -> None:
        y1 = self.h - 92
        y2 = self.h - 66
        cv2.rectangle(frame, (10, y1), (self.w - 10, y2), cfg.COLOR_HUD_BG, -1)
        color = _LEVEL_COLORS[d.longitudinal_level]
        cv2.rectangle(frame, (10, y1), (18, y2), color, -1)   # colour tab
        text = self._fit_text(d.reason, self.w - 40, 0.5)
        cv2.putText(frame, text, (26, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)

    @staticmethod
    def _fit_text(text: str, max_w: int, scale: float) -> str:
        """Truncate text with an ellipsis so it fits within max_w pixels."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        if cv2.getTextSize(text, font, scale, 1)[0][0] <= max_w:
            return text
        while len(text) > 4 and cv2.getTextSize(text + "...", font, scale, 1)[0][0] > max_w:
            text = text[:-1]
        return text + "..."
