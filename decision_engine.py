"""
decision_engine.py — Module 3: the decision engine ("the brain").

Modules 1, 2 and 4 are the eyes and memory. This module is the brain. It never
touches pixels for perception — it reasons over the tracked objects (Module 4)
and the lane result (Module 1) and fuses them into ONE arbitrated driving action
per frame, plus a single human-readable reason a driver or a log would understand.

How it works (end to end, once per frame):

    1. Trust check        — set valid / degraded flags from the inputs.
    2. Hazard selection    — pick the nearest CONFIRMED, non-advisory, in-path
                            track. Its per-object smoothed distance / closing
                            speed / TTC come straight from the tracker, so a
                            hazard's kinematics stay stable even in dense traffic
                            (no re-seeding when the closest object changes).
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
lane path can never weaken the longitudinal decision. Only *confirmed* tracks
influence decisions, so a one-frame false detection can never move the car.
"""

import logging
from collections import deque
from dataclasses import dataclass, replace
from typing import List, Optional, Tuple

import cv2

import config as cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Longitudinal levels
# ---------------------------------------------------------------------------

PROCEED, CAUTION, SLOW, BRAKE, EMERGENCY_STOP = 0, 1, 2, 3, 4
LEVEL_NAMES = ["PROCEED", "CAUTION", "SLOW", "BRAKE", "EMERGENCY_STOP"]
_RISK_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}

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
    hazard_id:          Optional[int]   = None   # tracker ID of the hazard
    smoothed_distance_m: Optional[float] = None
    closing_speed_mps:  Optional[float] = None
    ttc_s:              Optional[float] = None
    hazard_side:        Optional[str]   = None   # 'LEFT' / 'RIGHT' / None
    tracked_count:      int   = 0
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
        decision = engine.process(lane_result, tracks)   # per frame
        annotated = engine.draw_hud(annotated, decision)

    ``tracks`` is the confirmed+tentative Track list from Module 4 (or None if
    object tracking is disabled). The engine only acts on confirmed tracks.
    """

    def __init__(self, frame_width: int, frame_height: int) -> None:
        self.w = frame_width
        self.h = frame_height

        # ── Committed-level ratchet state ─────────────────────────────
        self.committed_level = PROCEED
        self.raw_history: "deque[int]" = deque(maxlen=cfg.EVIDENCE_WINDOW)
        self.down_counter = 0
        self.emergency_dwell = 0

        # ── Timing / bookkeeping ──────────────────────────────────────
        self._prev_tick = cv2.getTickCount()
        self.frame_index = 0
        self._last_decision: Optional[DrivingDecision] = None

        logger.info(f"DecisionEngine initialised ({frame_width}x{frame_height})")

    def reset(self) -> None:
        """Clear temporal state — call when a video loops back to the start."""
        self.committed_level = PROCEED
        self.raw_history.clear()
        self.down_counter = 0
        self.emergency_dwell = 0
        self._last_decision = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, lane_result, tracks) -> DrivingDecision:
        """
        Fuse one frame of lane result + object tracks into a driving decision.

        Either argument may be None (its module was disabled or unavailable);
        the engine degrades gracefully. If BOTH are None it holds the previous
        decision rather than inventing one.
        """
        self.frame_index += 1
        raw_dt = self._measure_dt()             # only used for the fps trust gate

        # ── Validity: nothing to reason about ─────────────────────────
        if lane_result is None and tracks is None:
            return self._hold_or_default()

        lane_conf = lane_result.confidence if lane_result is not None else 0.0
        fps = 1.0 / raw_dt if raw_dt > 1e-6 else 999.0

        # ── Only confirmed tracks influence decisions ─────────────────
        confirmed = [t for t in (tracks or []) if t.confirmed]
        hazard = self._nearest_actionable(confirmed)
        highest_risk = self._highest_risk(confirmed)
        hazard_side = self._hazard_side(hazard)

        # ── Trust / degraded flags ────────────────────────────────────
        degraded, degraded_cause = self._assess_trust(lane_result, tracks, hazard, fps)

        # ── Policy table -> raw longitudinal level ────────────────────
        raw_level, rule_id, reason_core = self._evaluate_rules(
            hazard, degraded, confirmed, highest_risk
        )

        # ── Temporal ratchet: raw -> committed ────────────────────────
        committed = self._ratchet(raw_level, degraded)

        # ── Degraded floor: never claim PROCEED while blind + hazard ──
        floored = False
        floor_reason = ""
        if degraded and committed < CAUTION and (
            hazard is not None or highest_risk in ("MEDIUM", "HIGH")
        ):
            committed = CAUTION
            floored = True
            floor_reason = "inputs untrusted with a hazard nearby"

        # ── Vulnerable-proximity floor: never claim PROCEED while a confirmed
        #    vulnerable road user is in the ego path within close range. The
        #    ratchet takes a frame or two to confirm a newly-entered hazard
        #    (e.g. a pedestrian/rider crossing into the corridor); for a close
        #    VRU even one frame of PROCEED is unacceptable, so floor it here.
        #    One-directional (only ever raises caution), independent of the
        #    degraded flag, and gated on confirmed + in_path so it can't fire on
        #    a roadside bystander. Mirrors the spine audit invariant exactly.
        if committed < CAUTION:
            vru = next(
                (t for t in confirmed
                 if t.label in cfg.VULNERABLE_CLASSES and getattr(t, "in_path", False)
                 and t.smoothed_distance_m is not None
                 and t.smoothed_distance_m <= cfg.VULNERABLE_FLOOR_DIST_M),
                None,
            )
            if vru is not None:
                committed = CAUTION
                floored = True
                floor_reason = f"{vru.label} #{vru.id} {vru.smoothed_distance_m:.1f}m in-path"

        self.committed_level = committed
        self.emergency_dwell = self.emergency_dwell + 1 if committed == EMERGENCY_STOP else 0

        # ── Lateral action, safety-arbitrated ─────────────────────────
        lateral, lateral_mag, lateral_note = self._lateral(lane_result, committed, hazard_side)

        # ── Command scalars ───────────────────────────────────────────
        ttc = hazard.ttc_s if hazard is not None else None
        throttle, brake = self._scalars(committed, ttc)

        # ── Assemble the reason string (honest about WHY committed level) ──
        core_text = reason_core.split("] ", 1)[-1]
        if floored:
            rid = "FLOOR"
            reason = f"[FLOOR] CAUTION: easing off -- {floor_reason}"
        elif committed == raw_level:
            rid = rule_id
            reason = reason_core
        elif committed > raw_level:
            rid = rule_id
            reason = f"[{rule_id}] {LEVEL_NAMES[committed]}: holding -- live read {core_text}"
        else:
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
            hazard_id=hazard.id if hazard is not None else None,
            smoothed_distance_m=(round(hazard.smoothed_distance_m, 1)
                                 if hazard is not None and hazard.smoothed_distance_m is not None
                                 else None),
            closing_speed_mps=round(hazard.closing_speed_mps, 1) if hazard is not None else 0.0,
            ttc_s=round(ttc, 1) if ttc is not None else None,
            hazard_side=hazard_side,
            tracked_count=len(confirmed),
            lane_confidence=lane_conf,
            frame_index=self.frame_index,
            fps=fps,
        )
        self._last_decision = decision
        return decision

    # ------------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------------

    def _measure_dt(self) -> float:
        """
        Unclamped seconds since the previous call, for the fps trust gate — a
        genuine stall must be visible (a clamp would floor fps at MIN_FPS and
        hide it). The per-object kinematics dt lives in the tracker.
        """
        tick = cv2.getTickCount()
        dt = (tick - self._prev_tick) / cv2.getTickFrequency()
        self._prev_tick = tick
        return dt

    # ------------------------------------------------------------------
    # Hazard selection over tracks
    # ------------------------------------------------------------------

    def _nearest_actionable(self, tracks):
        """
        Nearest CONFIRMED, non-advisory, in-path track — the object the
        collision rules act on. Advisory signs (handled by R4) are excluded so
        a foreground sign can't mask a closing vehicle behind it.
        """
        candidates = [
            t for t in tracks
            if t.in_path and t.label not in cfg.ADVISORY_CLASSES
            and t.smoothed_distance_m is not None
        ]
        return min(candidates, key=lambda t: t.smoothed_distance_m) if candidates else None

    def _highest_risk(self, tracks) -> str:
        if not tracks:
            return "LOW"
        return max((t.risk for t in tracks), key=lambda r: _RISK_RANK.get(r, 0))

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
    # Trust / degraded assessment
    # ------------------------------------------------------------------

    def _assess_trust(self, lane_result, tracks, hazard, fps) -> Tuple[bool, str]:
        """Returns (degraded, first_cause_string)."""
        if lane_result is None:
            return True, "object-only mode"
        if lane_result.confidence == 0.0:
            return True, "no lane lock -- corridor assumed frame-centre"
        if tracks is None:
            return True, "lane-only mode"
        if fps < cfg.MIN_FPS:
            return True, f"low fps ({fps:.1f})"
        if hazard is not None and hazard.confidence < cfg.DET_CONF_MIN:
            return True, "low-confidence detection"
        return False, ""

    # ------------------------------------------------------------------
    # Policy table  (ordered, first-match-wins -> raw longitudinal level)
    # ------------------------------------------------------------------

    def _evaluate_rules(self, hazard, degraded, tracks, highest_risk) -> Tuple[int, str, str]:
        # ── Rules needing the nearest actionable hazard (R1..R3) ──────
        if hazard is not None:
            d = hazard.smoothed_distance_m
            ttc = hazard.ttc_s
            closing = hazard.closing_speed_mps > cfg.MIN_CLOSING_MPS
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
                    r = f"[R1] EMERGENCY_STOP: {label} #{hazard.id} {d:.1f}m in-path, TTC {ttc:.1f}s closing"
                else:
                    r = f"[R1] EMERGENCY_STOP: {label} #{hazard.id} {d:.1f}m in-path"
                return EMERGENCY_STOP, "R1", r

            # R2 — BRAKE (closing fast)
            if ttc is not None and ttc <= ttc_brake and closing:
                vuln = " (vulnerable, early brake)" if label in cfg.VULNERABLE_CLASSES else ""
                return BRAKE, "R2", f"[R2] BRAKE: {label} #{hazard.id} closing, {d:.1f}m TTC {ttc:.1f}s{vuln}"

            # R3 — BRAKE (static HIGH-risk object dead ahead)
            if hazard.risk == "HIGH" and d <= cfg.RISK_DISTANCE_HIGH:
                return BRAKE, "R3", f"[R3] BRAKE: {label} #{hazard.id} {d:.1f}m in-path (HIGH)"

        # ── R4 — SLOW for an in-path traffic control (advisory ease) ──
        # We only EASE, never stop hard: Module 2 reports no light colour.
        advisory_hit = None
        for t in tracks:
            if (t.in_path and t.label in cfg.ADVISORY_CLASSES
                    and t.smoothed_distance_m is not None
                    and t.smoothed_distance_m <= cfg.STOPSIGN_DISTANCE_M):
                if advisory_hit is None or t.smoothed_distance_m < advisory_hit.smoothed_distance_m:
                    advisory_hit = t
        if advisory_hit is not None:
            return SLOW, "R4", f"[R4] SLOW: {advisory_hit.label} {advisory_hit.smoothed_distance_m:.0f}m ahead"

        # ── R5 — SLOW for a MEDIUM-risk in-path object ────────────────
        if hazard is not None:
            margin = cfg.DEGRADED_DIST_MARGIN_M if degraded else 0.0
            if hazard.risk == "MEDIUM" and hazard.smoothed_distance_m <= cfg.RISK_DISTANCE_MEDIUM + margin:
                return SLOW, "R5", f"[R5] SLOW: {hazard.label} #{hazard.id} {hazard.smoothed_distance_m:.1f}m in-path (MEDIUM)"

        # ── R6 — CAUTION (gentle in-path closing, or any nearby risk) ─
        if hazard is not None and hazard.ttc_s is not None and hazard.closing_speed_mps > cfg.MIN_CLOSING_MPS:
            ttc_margin = cfg.DEGRADED_TTC_MARGIN_S if degraded else 0.0
            if hazard.label in cfg.VULNERABLE_CLASSES:
                ttc_margin += cfg.VULNERABLE_TTC_MARGIN_S
            if (cfg.TTC_BRAKE_S + ttc_margin) < hazard.ttc_s <= (cfg.TTC_CAUTION_S + ttc_margin):
                return CAUTION, "R6", f"[R6] CAUTION: {hazard.label} #{hazard.id} closing gently, TTC {hazard.ttc_s:.1f}s"
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
            # Any hotter-than-committed read breaks the calm streak, even if not
            # yet confirmed enough to escalate.
            self.down_counter = 0
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
        if lane_result is None or lane_result.confidence == 0.0:
            return "HOLD", "NONE", None

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
        if committed == EMERGENCY_STOP:
            return "HOLD", "NONE", "straight-line braking"

        if committed >= BRAKE and action != "KEEP_LANE":
            toward = ((action == "CORRECT_LEFT" and hazard_side == "LEFT")
                      or (action == "CORRECT_RIGHT" and hazard_side == "RIGHT"))
            if toward:
                return "HOLD", "NONE", "lateral inhibited: correction toward hazard"

        if committed >= SLOW and mag in ("MODERATE", "HARD"):
            mag = "SLIGHT"

        return action, mag, note

    # ------------------------------------------------------------------
    # Command scalars
    # ------------------------------------------------------------------

    def _scalars(self, level: int, ttc: Optional[float]) -> Tuple[float, float]:
        """Map a committed level to (throttle, brake) in 0.0-1.0."""
        if level == PROCEED:
            return 1.0, 0.0
        if level == CAUTION:
            return 0.0, cfg.BRAKE_EASE
        if level == SLOW:
            return 0.0, cfg.BRAKE_SLOW
        if level == BRAKE:
            if ttc is not None and ttc > 0:
                b = max(cfg.BRAKE_MIN, min(cfg.BRAKE_MAX, 1.0 - ttc / cfg.TTC_BRAKE_S))
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
        # centred Forward Collision Warning banner and hide it — drop it below.
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

        # ── Nearest-in-path summary (with tracker ID) ─────────────────
        if d.hazard_label:
            dist = f"{d.smoothed_distance_m:.0f}m" if d.smoothed_distance_m is not None else "--"
            ttc = f"{d.ttc_s:.1f}s" if d.ttc_s is not None else "--"
            trend = "^" if (d.closing_speed_mps or 0) > cfg.MIN_CLOSING_MPS else "v"
            ident = f"#{d.hazard_id} " if d.hazard_id is not None else ""
            ahead = f"AHEAD {ident}{d.hazard_label} {dist} TTC {ttc} {trend}"
        else:
            ahead = "AHEAD  clear"
        cv2.putText(frame, ahead, (x0 + 12, y0 + 106),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Telemetry line (closing speed, fps, tracked count) ────────
        v = d.closing_speed_mps if d.closing_speed_mps is not None else 0.0
        cv2.putText(frame, f"v {v:+.1f} m/s  fps {d.fps:.0f}  trk {d.tracked_count}",
                    (x0 + 12, y0 + 128), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1, cv2.LINE_AA)

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
