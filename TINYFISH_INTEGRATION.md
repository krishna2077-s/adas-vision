# TinyFish AI — ADAS Route Hazard & Environmental Scout (Module 15)

This module integrates **TinyFish AI** (`agent.tinyfish.ai`) into **ADAS Vision** as an advisory web perception layer. It autonomously scouts live road conditions, severe weather warnings, construction bottlenecks, and traffic incidents ahead on the vehicle's route, projecting live advisory chips onto the HUD.

---

## 1. Architectural Highlights

- **Decoupled & Non-Hardcoded**: Controlled via the `ENABLE_TINYFISH` feature flag and `TINYFISH_API_KEY` environment variable. If disabled or unset, the module stays dormant or uses realistic local fallback simulation.
- **Credit-Preserving**: By default, routine queries use TinyFish's **Free Search Gateway** (12,000 free requests/day), protecting your \$15 credit balance. The metered Agent Automation engine (\$0.016/step) is only activated when explicitly requested with `--deep`.
- **Asynchronous & Non-Blocking**: Web queries execute on an isolated background worker thread, ensuring the core video perception and decision engine pipelines maintain full 30–60 FPS performance without I/O stalls.
- **Advisory-Only Safety**: Conforms to the ADAS Vision safety invariants — it generates informative HUD alerts and recommended advisory speeds, but never directly actuates steering, throttle, or brakes.

---

## 2. Quickstart & Verification

### Option A: Standalone CLI (Zero Video Dependencies)
Run a quick live route scout from your terminal:

```bash
# 1. Offline / Mock Mode (0 API calls, 0 credits used)
python run.py scout --mock

# 2. Live Query using Free Search Tier (12k/day free, requires TINYFISH_API_KEY)
$env:TINYFISH_API_KEY="sk-tinyfish-YOUR_KEY"
python run.py scout --location "NH48 Delhi Gurgaon"

# 3. Deep Autonomous Agent Mode (Browses web with natural language goal, metered from credits)
python run.py scout --location "Mumbai Pune Expressway" --deep
```

### Option B: Video HUD Overlay Integration
Launch the main ADAS Vision pipeline with live TinyFish scouting enabled:

```bash
# Run on dashcam video with TinyFish scout overlay
$env:TINYFISH_API_KEY="sk-tinyfish-YOUR_KEY"
python run.py main --video dashcam.mp4 --tinyfish --tinyfish-location "NH48 Delhi-Gurgaon"
```

### Option C: Automated Unit Tests
Run the test suite to verify module isolation, thread safety, and fallback handling:

```bash
python evaluation/test_tinyfish_scout.py
```

---

## 3. Environment Variables

| Variable | Description | Default |
|---|---|---|
| `TINYFISH_API_KEY` | Your TinyFish API key from [agent.tinyfish.ai/api-keys](https://agent.tinyfish.ai/api-keys) | `""` |
| `ENABLE_TINYFISH` | Master toggle to enable Module 15 in the main pipeline | `false` |
| `TINYFISH_LOCATION` | Active highway or city route to scout | `"NH48 Delhi-Gurgaon"` |
| `TINYFISH_POLL_INTERVAL_S` | Interval in seconds between background web scans | `60.0` |

---

## 4. How to Remove After Your Hackathon

Because this entire implementation is contained within this dedicated git branch (`feature/tinyfish-integration`), removing it after your hackathon is effortless:

```bash
# 1. Switch back to your clean main branch
git checkout main

# 2. (Optional) Delete the feature branch permanently
git branch -D feature/tinyfish-integration
```

Your `main` branch remains 100% clean and untouched.
