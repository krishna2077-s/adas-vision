# ADAS Vision: Adaptive Path Planning and Collision Avoidance for Autonomous Vehicles on Unstructured Indian Roads

> **Smart India Hackathon / MathWorks Competition Submission**  
> **Team:** Krishna & Akshit (`krishna2077-s / akshit-mat`)  
> **Repository:** https://github.com/krishna2077-s/adas-vision  

---

## 1. Executive Summary

Autonomous navigation in developing countries faces challenges distinct from Western highway environments. Indian roads feature **unstructured geometry**, frequent **absence of painted lane markings**, **highly heterogeneous mixed traffic** (two-wheelers, auto-rickshaws, commercial trucks, pushcarts, and stray cattle), and **unpredictable actor trajectories**. Traditional autonomous pipelines that depend strictly on structured HD maps and constant-velocity lane-following fail catastrophically in these conditions.

We present **ADAS Vision**, an end-to-end, closed-loop autonomous driving architecture specifically engineered for unstructured Indian road conditions. Our system integrates:
1. **Multi-tier Perception Ladder**: Deep neural drivable area segmentation (LRASPP MobileNetV3 fine-tuned on the India Driving Dataset) + YOLOv8 + classical texture/chroma fallback for unpainted roads.
2. **Class-Aware Trajectory Prediction**: Social-force inspired motion forecasting applying class-dependent lateral uncertainty inflation ($\pm 2.0\text{m}$ for cattle, $\pm 1.5\text{m}$ for pedestrians).
3. **Dynamic Occupancy Grids & Real-Time Hybrid A* Replanning**: Vectorized $60\text{m} \times 30\text{m}$ spatial grid updating at $30\text{ Hz}$ with sub-millisecond ($0.04 - 0.2\text{ ms}$) kinematically feasible spline-smoothed path replanning.
4. **Deterministic R1–R7 Safety Spine with Temporal Ratchet**: Formally verifiable state arbitration featuring $N$-of-$M$ voting escalation, asymmetric de-escalation dampening, and a non-negotiable Vulnerable Road User (VRU) proximity safety floor.
5. **Non-linear Kinematic Bicycle Dynamics**: Closed-loop Pure Pursuit lateral tracking and PID longitudinal control.

---

## 2. System Architecture

The architecture connects perception, prediction, planning, and control in a unified closed loop:

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                1. SENSOR & PERCEPTION LAYER                             │
│  • Monocular Dashcam Video / Camera Stream                                              │
│  • Deep Drivable Surface Segmentation (IDD-trained LRASPP CNN)                          │
│  • YOLOv8 Multi-Class Detector (Pedestrians, Two-Wheelers, Cows, Rickshaws, Trucks)      │
│  • SORT-lite 2D Tracker with Inter-frame Kalman Kinematics                              │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ Tracks & Drivable Bounds
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                2. PREDICTION & WORLD MODEL                              │
│  • Constant Velocity + Class-Dependent Lateral Uncertainty Inflation                   │
│  • Dynamic 2D Binary Occupancy Grid ($60\text{m} \times 30\text{m}$, $\Delta=0.25\text{m}$)                │
│  • Stamp Current Footprints + Forecast Trajectory Cones                                 │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ Dynamic Grid + Predictions
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           3. DECISION ENGINE (Stateflow / R1–R7)                        │
│  • Hierarchical Rules: R1 (E-STOP) -> R2/R3 (BRAKE) -> R4/R5 (SLOW) -> R6 -> R7 (PROCEED) │
│  • $N$-of-$M$ Voting Temporal Ratchet & 15-frame Emergency Latch                        │
│  • VRU Proximity Safety Floor ($d \le 15\text{m} \implies \text{min CAUTION}$)          │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ Velocity Target + Hazard Bounds
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            4. ADAPTIVE PATH PLANNER (Hybrid A*)                         │
│  • Real-time Replan Trigger: Path-Prediction Intersection Check ($< 2.0\text{m}$ buffer) │
│  • Hybrid A* Kinematic Search on Binary Occupancy Map                                   │
│  • Cubic Spline Waypoint Smoothing ($\Delta s = 0.5\text{m}$)                            │
└────────────────────────────────────────────┬────────────────────────────────────────────┘
                                             │ Waypoints & Target Velocity
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              5. VEHICLE DYNAMICS & CONTROL                              │
│  • Pure Pursuit Steering ($\text{lookahead} = 6.0\text{m}$)                              │
│  • PID Longitudinal Throttle / Brake with Emergency Braking Override ($-7.0\text{ m/s}^2$)│
│  • Non-linear Kinematic Bicycle Dynamics Integration                                   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Mathematical & Algorithmic Formulation

### 3.1 Kinematic Bicycle Model
The vehicle state vector $\mathbf{x} = [x, y, \psi, v]^T$ evolves according to:
$$\dot{x} = v \cos(\psi + \beta)$$
$$\dot{y} = v \sin(\psi + \beta)$$
$$\dot{\psi} = \frac{v}{L} \tan(\delta) \cos(\beta)$$
$$\dot{v} = a$$
where $L = 2.70\text{ m}$ is the wheelbase, $\delta$ is the steering angle ($|\delta| \le 35^\circ$), and side-slip angle $\beta = \arctan\left(\frac{l_r}{L} \tan(\delta)\right)$.

### 3.2 Pure Pursuit Lateral Steering
Given a lookahead distance $L_a = 6.0\text{ m}$ and goal waypoint $(x_g, y_g)$ along the Hybrid A* spline:
$$\alpha = \text{atan2}(y_g - y, x_g - x) - \psi$$
$$\delta = \text{atan2}\left(\frac{2 L \sin(\alpha)}{L_a}\right)$$

### 3.3 Class-Dependent Trajectory Inflation
For tracked actor $i$ at $(x_i, y_i)$ with velocity $(v_{x,i}, v_{y,i})$ over prediction horizon $T = 3.0\text{ s}$:
$$x_i(t) = x_i + v_{x,i} t, \quad y_i(t) = y_i + v_{y,i} t$$
Uncertainty covariance inflation $\sigma_y(\text{class})$:
$$\sigma_y = \begin{cases} 
2.0\text{ m} & \text{class} \in \{\text{cow, animal}\} \\
1.5\text{ m} & \text{class} \in \{\text{pedestrian, VRU}\} \\
1.0\text{ m} & \text{class} \in \{\text{motorcycle, bicycle}\} \\
0.5\text{ m} & \text{class} \in \{\text{car, truck, bus}\}
\end{cases}$$

### 3.4 Replan Trigger Condition
Replanning is triggered instantaneously if:
$$t - t_{\text{last}} \ge 200\text{ ms} \quad \lor \quad \min_{j, k} \|\mathbf{p}_{\text{path}}(s_j) - \mathbf{p}_{\text{pred}, k}(t)\| < 2.0\text{ m} \quad \lor \quad \mathcal{M}(\mathbf{p}_{\text{path}}) = 1$$

---

## 4. Benchmark Validation Across 5 Mandatory Scenarios

All 5 scenarios were evaluated using automated benchmark scripts (`run_all_scenarios.m` and `run_baseline_vs_adaptive.m`):

| Scenario | Mode | Duration (s) | Path Roughness ($\text{rad/m}$) | Min Clearance (m) | Collisions | Status |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **1. Village Road** | Baseline | 24.0 | 0.0018 | 1.62 | 0 | PASS |
| | **Adaptive (Ours)** | **23.8** | **0.0009** | **3.60** | **0** | **PASS** |
| **2. Urban Intersection** | Baseline | 26.2 | 0.0022 | 1.10 | 0 | PASS |
| | **Adaptive (Ours)** | **25.3** | **0.0011** | **2.45** | **0** | **PASS** |
| **3. Highway Merge** | Baseline | 19.5 | 0.0015 | 1.20 | 0 | PASS |
| | **Adaptive (Ours)** | **18.9** | **0.0006** | **2.80** | **0** | **PASS** |
| **4. Dense Market** | Baseline | 32.0 | 0.0035 | 0.20 | 1 | FAIL (Hit) |
| | **Adaptive (Ours)** | **28.4** | **0.0003** | **0.44** | **0** | **PASS** |
| **5. Cattle Crossing** | Baseline | 31.0 | 0.0028 | 1.80 | 0 | PASS |
| | **Adaptive (Ours)** | **30.5** | **0.0008** | **3.90** | **0** | **PASS** |

### Key Experimental Insights:
* **True Clearance Doubled**: Adaptive Hybrid A* proactively shifted the vehicle away from static thelas and oncoming bikes, maintaining over **$2\times$ the physical clearance** compared to traditional lane-locked baseline ADAS.
* **Zero Collisions in Market Corridor**: In the congested $3.5\text{m}$ bazaar, traditional baseline clipped obstacles due to lack of lateral deviation, whereas our adaptive replanner successfully cleared the passage with zero collisions.
* **Sub-Millisecond Replan Latency**: Mean replanning latency across 700+ cycles was **$0.04 - 0.2\text{ ms}$**, verifying real-time embedded feasibility.

---

## 5. Demonstration Video Presentation Script (3–5 Minutes)

* **[0:00 - 0:45] Introduction & The Problem**: Show dashcam footage of chaotic Indian road with missing lane paint, stray cattle, and auto-rickshaws. Explain why standard lane-keeping ADAS fails.
* **[0:45 - 1:45] AI Vision & Drivable Segmentation**: Demonstrate live Python perception running on video. Show bounding boxes with TTC and confidence on unpainted roads.
* **[1:45 - 3:15] Closed-Loop Scenarios & Replan Visualization**: 
  * Show Scenario 5 (Cattle Crossing) stopping smoothly at $11.4\text{m}$ buffer distance.
  * Show Scenario 4 (Dense Market) navigating narrow $3.5\text{m}$ bazaar with active Hybrid A* spline adjustments.
  * Show cyan path ribbon and red predicted trajectory cones.
* **[3:15 - 4:15] Benchmark Results & Stateflow Verification**: Display the comparative performance table proving double clearance and zero collisions.
* **[4:15 - 4:45] Conclusion**: Summary of the Level 2+/3 closed-loop architecture.
