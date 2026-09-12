# 🏋️ AI-Powered Push-Up Counter

**Real-Time Pose Detection & Exercise Form Analysis**

An intelligent fitness application that watches you train, understands *how* your
body moves, counts only the repetitions that actually count, and coaches your
form in real time — powered by **RF-DETR pose estimation**, a **joint-angle state
machine**, and a **weighted form-scoring engine**.

> This is not a "count the pixels that moved" demo. The system extracts 17 body
> keypoints, computes real joint angles, runs a debounced finite-state machine
> over the elbow angle, validates every repetition against posture rules, and
> scores your technique 0–100.

---

## ✨ Features

| Area | What it does |
| --- | --- |
| **Pose detection** | RF-DETR Keypoint (COCO 17) with a confidence gate; dependency-free **demo** backend for zero-setup runs |
| **Joint angles** | Left/right elbow + body-alignment angles via vector geometry, NaN-safe |
| **Rep counting** | Finite-state machine `UP → MOVING → DOWN → MOVING → UP`; only full, valid cycles count |
| **Invalid-rep filtering** | Rejects shallow, too-fast, misaligned, uneven or tracking-lost attempts — with a reason |
| **Form scoring** | Weighted 0–100 score (depth · alignment · symmetry · control) with grade bands |
| **Live coaching** | Real-time cues: *"Don't raise your hips"*, *"Lower your chest more"*, *"Slow down"* |
| **Pro overlay** | Skeleton, angle arcs, rep counter, status bar, form bar, telemetry — rendered with OpenCV |
| **Designed dashboard** | "PULSE·REP" dark-athletic UI: KPI cards, SVG form ring, angle meters, live status pills, themed sidebar (design system in `src/ui/`) |
| **3 input modes** | Live webcam · uploaded video · built-in demo stream |
| **Transport controls** | Start / Stop / Pause / Resume / Step-frame / Screenshot / Reset / Export |
| **Video export** | Writes the fully annotated video to **MP4, AVI, MKV, MOV or animated GIF** |
| **Real-time preview** | Live/demo playback paced to the source fps (toggleable) for a true real-life feel |
| **Analytics** | Interactive Plotly charts + JSON/CSV/Markdown workout reports |
| **Multi-person** | Picks and *stably tracks* one primary subject (confidence · size · centrality) |
| **Optional REST API** | FastAPI server for headless/image/video analysis |

---

## 🎬 Demo

The repo ships **six synthetic demo clips** (plus an animated GIF preview) under
`assets/demo/`, so you can run the **entire pipeline with no camera and no model
weights** — and see the form feedback react to deliberate faults:

| Clip | What it shows |
|------|---------------|
| `demo_pushup.mp4` | Clean, well-paced reps (the "reference" form) |
| `demo_fast.mp4` | Reps that are too fast (timing/validation faults) |
| `demo_hips_high.mp4` | Hips piking up (body-alignment fault) |
| `demo_hips_low.mp4` | Hips sagging down (body-alignment fault) |
| `demo_uneven_arms.mp4` | One arm ahead of the other (symmetry fault) |
| `demo_pushup.gif` | 640×360 animated preview of the clean clip |

```bash
streamlit run app.py        # pick "🎬 Demo Video" + "Demo (no weights)" engine
# or headless:
python cli.py --demo --engine demo
# inject a specific fault into the synthetic demo:
python cli.py --demo --engine demo --hips-offset 0.12     # hips piking
python cli.py --demo --engine demo --arm-asymmetry 30     # uneven arms
python cli.py --demo --engine demo --cycle 1.8            # faster tempo
```

Regenerate every clip and the GIF with `python -m scripts.make_demo_video`
(pass `build_all()`; each clip is written from `generate_pushup_keypoints`, so
the geometry stays consistent with the demo adapter used for analysis).

---

## 🏗️ Architecture

```
Camera / Video / Demo
        ↓
Frame Validation
        ↓
RF-DETR Pose Detection ──► Keypoint Extraction ──► Confidence Gate
        ↓
Primary-Person Selection (temporal tracking)
        ↓
Joint-Angle Engine ──► Angle Smoothing (MA / EMA)
        ↓
Movement Analyzer (velocity, phase, debounce)
        ↓
Posture Analyzer ──► Push-Up State Machine ──► Rep Validation
        ↓
Form Score Engine
        ↓
Skeleton Drawing ──► HUD Overlay ──► Analytics Tracking
        ↓
Display / Screenshot / Video Export / Report
```

**Adapter pattern:** the rest of the app only ever sees the standardized
`PersonPose` contract. Swapping RF-DETR for another estimator means writing one
adapter — nothing downstream changes.

---

## 📦 Installation

Requires **Python 3.11+**.

```bash
git clone <your-fork-url> AI_Pushup_Counter
cd AI_Pushup_Counter

python -m venv .venv
```

**Windows**
```powershell
.venv\Scripts\activate
```

**macOS / Linux**
```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

> **CPU-only PyTorch?** `rfdetr` pulls in `torch`. To avoid the multi-GB CUDA
> build, install CPU torch first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```

---

## ⚙️ Setup

1. **Model weights** — on first RF-DETR run the COCO keypoint checkpoint
   (`rf-detr-keypoint-preview-xlarge.pth`) is downloaded and cached automatically.
   To use a local checkpoint, set `PUSHUP_PRETRAIN_WEIGHTS` in `.env`.
2. **Environment overrides (optional)** — copy the template and edit:
   ```bash
   cp .env.example .env
   ```
   Precedence: **env vars → `config/config.yaml` → code defaults**.
3. **Want zero downloads?** Set `PUSHUP_ENGINE=demo` (or choose *Demo* in the
   sidebar). The synthetic backend produces the exact same keypoint contract.

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Then open the printed URL (default <http://localhost:8501>).

**Other entry points**

```bash
# Headless batch analysis of a video (writes video + reports)
python cli.py --source assets/demo/demo_pushup.mp4

# Zero-setup demo run
python cli.py --demo --engine demo

# Choose the export container (mp4 | avi | gif)
python cli.py --source assets/demo/demo_pushup.mp4 --format gif
python cli.py --demo --engine demo --format avi

# Optional REST API
pip install "fastapi>=0.110" "uvicorn[standard]>=0.29" python-multipart
uvicorn api_server:app --reload
```

---

## 🧠 How It Works

1. **Pose Detection** — RF-DETR predicts 17 COCO keypoints `(x, y, confidence)`
   per person in one forward pass.
2. **Keypoints** — normalized into a model-agnostic `PersonPose`; joints below the
   confidence threshold are marked invalid.
3. **Angle Calculation** — the angle at joint *B* from points *A, B, C*:
   ```
   BA = A - B,  BC = C - B
   angle = arccos( dot(BA, BC) / (|BA| · |BC|) )   → degrees
   ```
   Missing points, zero vectors and non-finite coords return `NaN` (never raise).
4. **Smoothing** — a moving average (or EMA) over the last *N* elbow angles kills
   pose jitter before classification.
5. **Movement Analysis** — the smoothed angle is bucketed into `UP / MOVING /
   DOWN`, **debounced** so a single noisy frame can't flip the state.
6. **State Machine** — a repetition counts only on a confirmed
   `UP → DOWN → UP` cycle where `reached_down` was set on the way down.
7. **Rep Validation** — each cycle is checked for depth, elbow travel, tempo,
   alignment, arm symmetry and tracking ratio; failures are logged with a reason.
8. **Form Analysis** — depth, alignment, symmetry and control are scored and
   weighted into a single 0–100 form score with a grade.

---

## 🎨 UI / UX Design

The dashboard (**PULSE·REP**) ships with a self-contained design system in `src/ui/`:

- **`src/ui/theme.py`** — single source of truth for the web surface: palette
  tokens (deep-navy surfaces, electric-lime + cyan accents, semantic
  good/warn/bad), radii, shadows, and the global stylesheet (gradient hero
  wordmark, glass cards, KPI grid, meters, status pills, styled transport
  buttons, themed sidebar, hidden stock Streamlit chrome).
- **`src/ui/components.py`** — pure HTML/SVG component builders (unit-tested,
  no external assets so they render anywhere): brand logo, hero with live
  status pill, KPI cards, an SVG form-score donut ring, joint-angle meters,
  coaching feedback cards and the footer.
- **`.streamlit/config.toml`** — aligns native widgets (sliders, toggles,
  selects) with the dark brand theme.

Page flow: hero + live status → transport toolbar → KPI row → *video feed*
beside a *live form panel* (score ring, state pill, angle meters, coaching) →
tabs for **Analytics / Reports & Downloads / Screenshots / Developer**.

Colour centralisation rule: the OpenCV overlay reads colours from
`config/config.yaml`; the web UI reads them from `src/ui/theme.py` — exactly
one place per surface, no hard-coded hex values in `app.py`.

---

## 📁 Project Structure

```
AI_Pushup_Counter/
├── app.py                      # Streamlit dashboard (layout + wiring only)
├── cli.py                      # Headless batch runner
├── api_server.py               # Optional FastAPI backend
├── requirements.txt
├── pyproject.toml              # pytest / ruff / mypy config
├── .streamlit/config.toml      # dashboard theme (dark base, brand accents)
├── .env.example
├── .gitignore
├── README.md
│
├── config/
│   ├── settings.py             # Typed Settings (YAML + env overrides)
│   └── config.yaml             # Master configuration
│
├── src/
│   ├── pose/                   # RF-DETR adapter, demo adapter, keypoints,
│   │                           #   PoseDetector facade, subject tracker
│   ├── analysis/               # angles, smoothing, movement, posture, form score
│   ├── counter/                # push-up state machine + rep validator
│   ├── visualization/          # colors, drawing primitives, skeleton, HUD overlay
│   ├── video/                  # sources, pipeline, exporter, session controller
│   ├── analytics/              # session tracker, Plotly charts, reports
│   ├── ui/                     # web design system: theme tokens + HTML/SVG components
│   └── utils/                  # logging + helpers
│
├── outputs/                    # screenshots/ videos/ reports/ (git-ignored)
├── tests/                      # pytest suite (angles, counter, form, pipeline, adapter, UI)
├── scripts/make_demo_video.py  # regenerates the bundled demo clips + GIF
└── assets/demo/                # demo_pushup / fast / hips_high / hips_low / uneven_arms + GIF
```

---

## 🔧 Configuration

Everything is tunable from `config/config.yaml` (or the sidebar / env vars):

```yaml
model:
  engine: "rf-detr"              # or "demo"
  confidence_threshold: 0.5

pushup:
  up_threshold: 150              # elbow angle ≥ this => UP
  down_threshold: 90             # elbow angle ≤ this => DOWN
  stable_frames: 3               # frames required to confirm a state change
  min_rep_duration: 0.55         # seconds; faster => rejected as noise

smoothing:
  enabled: true
  method: "moving_average"       # or "ema"
  window_size: 5

form:
  weights: {depth: 30, alignment: 30, symmetry: 20, control: 20}
  bands: {excellent: 90, good: 75, needs_improvement: 60}

video:
  output_fps: 30
  save_processed_video: true
  export_format: "mp4"           # mp4 | avi | mkv | mov | gif
  realtime_pacing: true          # pace live/demo preview to source fps
  display_width: 960             # UI preview downscale (export stays full-res)
  gif_width: 480                 # GIF exports are downscaled to this width
```

Colours, HUD layout, validation rules and analytics are all in the same file —
no magic numbers scattered through the code.

---

## 🧪 Testing

```bash
pytest -q
```

The suite (**82 tests**) covers the angle math (90°/180°/invalid points), the
counter FSM (full cycle = 1 rep, partial = 0, no duplicate counts), form analysis
(good/poor/uneven), the RF-DETR output adapter, multi-format export (MP4/AVI/GIF
round-trips and container fallback), the fault-injected demo source (hip pike,
arm asymmetry, configurable tempo), and full-pipeline + session integration — all
without needing torch or a camera.

---

## 🚀 Future Improvements

- [ ] Squat, pull-up and sit-up detection (reuse the angle/state-machine core)
- [ ] Automatic exercise recognition
- [ ] Multi-exercise sessions with per-move scoring
- [ ] Personalized AI coaching (LLM-generated cue summaries)
- [ ] Workout history database + progress trends
- [ ] User authentication and multi-user profiles
- [ ] Cloud deployment (Docker + hosted inference)
- [ ] ONNX/TensorRT export for edge devices

---

## 📄 License

MIT — see the license header in the source files. RF-DETR core models are
Apache-2.0 (Roboflow).

---

**Built to demonstrate:** Computer Vision · RF-DETR Pose Estimation · Keypoint
Analysis · Geometry & Joint Angles · Finite-State Machines · Real-Time Video
Processing · Exercise Form Analysis · AI Feedback · Data Visualization ·
Streamlit Application Development.
