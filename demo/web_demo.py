"""
web_demo.py — Browser UI for ADAS Vision (Gradio).

A local web app for *inspecting and debugging* the perception pipeline frame by
frame. It does not replace main.py (the live OpenCV app) — it's a shared visual
surface: you (and an assistant like Claude) can open the same localhost page,
scrub to any frame, see which guidance source won, and diagnose problems.

Run:
    python web_demo.py
Then open http://127.0.0.1:7860 in a browser.

Tabs:
    - Frame explorer : slider over dashcam.mp4 -> full guidance overlay +
                       diagnostics (each frame judged independently, so paint
                       ghost-memory doesn't leak between non-consecutive frames).
    - Upload image   : drop any road photo -> the learned drivable-area mask.

Advisory / research only.
"""

import os
import cv2
import numpy as np
import torch
import gradio as gr

# Run from the repo root so relative paths (dashcam.mp4, the .pth) resolve.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from lane_detection import LaneDetector
from road_detection import RoadDetector
from learned_road_detection import LearnedRoadDetector

try:
    from object_detection import ObjectDetector
    _HAS_OBJ = True
except Exception as _exc:          # ultralytics missing, etc.
    _HAS_OBJ = False
    print("object detection unavailable:", _exc)

try:
    from tinyfish_scout import TinyFishRoadScout, RoadHazardAlert
    _HAS_TINYFISH = True
except Exception as _exc:
    _HAS_TINYFISH = False
    print("tinyfish scout unavailable:", _exc)

VIDEO = "dashcam.mp4"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

# ── Video metadata ─────────────────────────────────────────────────────────
_cap = cv2.VideoCapture(VIDEO)
HAS_VIDEO = _cap.isOpened()
N_FRAMES = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if HAS_VIDEO else 0
W = int(_cap.get(3)) if HAS_VIDEO else cfg.FRAME_WIDTH
H = int(_cap.get(4)) if HAS_VIDEO else cfg.FRAME_HEIGHT
_cap.release()

# ── Persistent, expensive singletons (model loaded once) ───────────────────
_learned = LearnedRoadDetector(W, H)
_obj = ObjectDetector(frame_width=W, frame_height=H) if _HAS_OBJ else None


def _read_frame(idx: int):
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def explore(frame_idx, show_objects):
    """Run the guidance pipeline on ONE frame (fresh lane state -> no ghosting)."""
    if not HAS_VIDEO:
        return None, "dashcam.mp4 not found in the repo root."
    frame = _read_frame(frame_idx)
    if frame is None:
        return None, f"could not read frame {int(frame_idx)}"

    lane = LaneDetector(W, H)          # fresh -> judged independently
    road = RoadDetector(W, H)
    _learned.reset()

    annotated = frame.copy()
    lane_result, annotated = lane.process(annotated)
    source = "PAINT (Module 1)" if lane_result.confidence > 0 else "NONE (no guidance)"

    if lane_result.confidence == 0.0:
        if _learned.available:
            rr, annotated = _learned.process(frame, draw_on=annotated)
            if rr.confidence > 0.0:
                lane_result, source = rr, "LEARNED (Module 1c)"
        if source.startswith("NONE"):
            rr, annotated = road.process(frame, draw_on=annotated)
            if rr.confidence > 0.0:
                lane_result, source = rr, "CLASSICAL (Module 1b)"

    if show_objects and _obj is not None:
        obj_result, annotated = _obj.process(annotated, lane_result.lane_center_x)
        n_obj = len(obj_result.detections)
    else:
        n_obj = "off"

    scout_str = "off"
    if _HAS_TINYFISH:
        try:
            scout = TinyFishRoadScout(location="NH48 Delhi-Gurgaon", mock_mode=True)
            annotated = scout.draw_chip(annotated)
            al = scout.alert
            if al:
                scout_str = f"{al.hazard_type} [{al.severity}] | Cap: {al.recommended_speed_kmh} km/h"
        except Exception:
            pass

    diag = (
        f"Frame          : {int(frame_idx)} / {N_FRAMES - 1}\n"
        f"Guidance source: {source}\n"
        f"Confidence     : {lane_result.confidence:.2f}\n"
        f"Steering       : {lane_result.steering}\n"
        f"Lane center x  : {lane_result.lane_center_x}\n"
        f"Offset (px)    : {lane_result.offset_px}\n"
        f"Objects        : {n_obj}\n"
        f"TinyFish Scout : {scout_str}\n"
        f"Learned model  : {'loaded' if _learned.available else 'MISSING (.pth not found)'}"
    )
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), diag


def segment_image(img_rgb):
    """Run just the learned drivable-area model on any uploaded image."""
    if img_rgb is None:
        return None, ""
    if not _learned.available:
        return img_rgb, "Learned model not loaded — place drivable_idd_full_best.pth in the repo root."
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = frame.shape[:2]
    inp = cv2.cvtColor(cv2.resize(frame, (cfg.LEARNED_INPUT_W, cfg.LEARNED_INPUT_H)), cv2.COLOR_BGR2RGB)
    x = ((inp.astype(np.float32) / 255.0 - _MEAN) / _STD).transpose(2, 0, 1)
    x = torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0)
    with torch.no_grad():
        pred = _learned.model(x)["out"].argmax(1)[0].numpy().astype(np.uint8)
    mask = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    fill = frame.copy()
    fill[mask > 0] = cfg.LEARNED_FILL_COLOR
    vis = cv2.addWeighted(frame, 1.0, fill, 0.45, 0)
    cov = float((mask > 0).mean())
    return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), f"Drivable-area coverage: {cov * 100:.1f}%"


def scout_ui(location, heading, km_marker, tier):
    """Interactively query TinyFish AI Scout and preview the HUD badge."""
    if not _HAS_TINYFISH:
        return None, {}, "TinyFish Scout module not available."
    is_deep = "Deep" in tier
    heading_val = None if heading == "Not specified" else heading
    scout = TinyFishRoadScout(
        location=location,
        heading=heading_val,
        km_marker=float(km_marker) if km_marker is not None else None,
        mock_mode=not bool(os.environ.get("TINYFISH_API_KEY")),
    )
    alerts = scout.scout_hazards(location, deep_agent=is_deep, force_refresh=True)
    alert = alerts[0] if alerts else None

    # Render HUD badge on sample 720p highway canvas
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
    canvas[:] = (35, 30, 28)
    cv2.putText(canvas, f"ROUTE: {location}  |  HEADING: {heading}  |  KM: {km_marker}",
                (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    annotated = scout.draw_chip(canvas)

    payload = alert.to_dict() if alert else {}
    summary = (
        f"Hazard Type  : {alert.hazard_type}\n"
        f"Severity     : {alert.severity}\n"
        f"Safe Speed   : {alert.recommended_speed_kmh} km/h (Capped in Planner & Decision Engine)\n"
        f"Headline     : {alert.headline}\n"
        f"Source       : {alert.source}\n"
        f"Freshness    : {'STALE (>300s)' if alert.is_stale else 'FRESH (Live Scout Cache)'}"
    ) if alert else "No active hazards found."
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), payload, summary


with gr.Blocks(title="ADAS Vision — debug") as demo:
    gr.Markdown(
        "# ADAS Vision — perception debugger\n"
        "Scrub to any frame and see which guidance source wins "
        "(**paint → learned → classical**) plus live diagnostics. "
        "_Advisory / research only._"
    )

    with gr.Tab("Frame explorer"):
        with gr.Row():
            with gr.Column(scale=1):
                slider = gr.Slider(0, max(N_FRAMES - 1, 1), value=min(15000, max(N_FRAMES - 1, 1)),
                                   step=1, label="Frame")
                show_obj = gr.Checkbox(value=bool(_HAS_OBJ), label="Show object detection (YOLOv8n)")
                btn = gr.Button("Analyze frame", variant="primary")
                gr.Markdown("Try **15000** (unmarked hill road → learned) vs **2000** (marked highway → paint).")
            with gr.Column(scale=2):
                out_img = gr.Image(label="Pipeline output")
                out_txt = gr.Textbox(label="Diagnostics", lines=8)
        btn.click(explore, [slider, show_obj], [out_img, out_txt])
        slider.release(explore, [slider, show_obj], [out_img, out_txt])

    with gr.Tab("Upload image"):
        with gr.Row():
            up_in = gr.Image(label="Road photo", type="numpy")
            up_out = gr.Image(label="Learned drivable area")
        up_txt = gr.Textbox(label="Info", lines=1)
        up_in.change(segment_image, up_in, [up_out, up_txt])

    with gr.Tab("TinyFish AI Route Scout (Module 15)"):
        gr.Markdown(
            "### Live Environmental & Route Hazard Scout (Module 15)\n"
            "Powered by **TinyFish AI**. Multi-tier scouting: Free Search Gateway (Tier 1), "
            "Fetch Markdown Parser (Tier 1.5), and Autonomous Browser Agent (Tier 2).\n"
            "_Feeds closed-loop speed caps directly into the prediction planner and decision engine._"
        )
        with gr.Row():
            with gr.Column(scale=1):
                tf_loc = gr.Dropdown(
                    choices=[
                        "NH48 Delhi-Gurgaon Expressway",
                        "Mumbai-Pune Expressway",
                        "Outer Ring Road Bengaluru",
                        "NH44 Panipat Stretch",
                        "Yamuna Expressway",
                    ],
                    value="NH48 Delhi-Gurgaon Expressway",
                    label="Corridor / Location",
                )
                tf_heading = gr.Dropdown(
                    choices=["Southbound", "Northbound", "Eastbound", "Westbound", "Not specified"],
                    value="Southbound",
                    label="Corridor Heading",
                )
                tf_km = gr.Number(value=24.0, label="Kilometer Marker (KM)")
                tf_tier = gr.Radio(
                    choices=["Fast Search Gateway (Free, 0 credits)", "Deep Browser Agent ($0.016/step)"],
                    value="Fast Search Gateway (Free, 0 credits)",
                    label="Scouting Tier",
                )
                tf_btn = gr.Button("Scout Corridor Hazards", variant="primary")
            with gr.Column(scale=2):
                tf_preview = gr.Image(label="HUD Advisory Badge Preview")
                tf_json = gr.JSON(label="Structured RoadHazardAlert Payload")
                tf_status = gr.Textbox(label="Scout Summary & Advisory Speed", lines=6)

        tf_btn.click(scout_ui, [tf_loc, tf_heading, tf_km, tf_tier], [tf_preview, tf_json, tf_status])


if __name__ == "__main__":
    print(f"video: {'found' if HAS_VIDEO else 'MISSING'} ({N_FRAMES} frames, {W}x{H}) | "
          f"learned model: {'loaded' if _learned.available else 'MISSING'} | "
          f"objects: {'on' if _HAS_OBJ else 'off'}")
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
