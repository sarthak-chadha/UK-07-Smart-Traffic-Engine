# 🚦 UK-07 Smart Traffic Engine

> **GNN-powered real-time traffic prediction and smart routing for Dehradun–Mussoorie, Uttarakhand.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-red?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/Backend-HuggingFace%20Spaces-yellow?logo=huggingface)](https://huggingface.co/spaces/Sarthak2606/UK-07_Smart_Traffic_Engine)
[![PWA](https://img.shields.io/badge/Frontend-PWA-purple?logo=googlechrome)](./TerrainNav_App)

---

## 📌 Overview

The UK-07 Smart Traffic Engine is an end-to-end deep learning navigating system that:

- Trains a **Graph Attention Network (GATv2)** on live TomTom traffic data fused with OpenStreetMap road topology and SRTM elevation data.
- Serves real-time traffic-aware routing via a **FastAPI** backend deployed on HuggingFace Spaces.
- Delivers navigation through **TerrainNav**, a Vanilla JavaScript **Progressive Web App (PWA)** installable on Android/iOS directly from a browser.

The system is specifically engineered for **mountainous terrain** — GNN residual connections preserve elevation topology data to prevent over-smoothing across steep gradient road segments.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     TerrainNav PWA (Frontend)              │
│   index.html · app.js · map.js · api.js · config.js       │
│              Leaflet.js + TomTom Maps SDK                  │
└───────────────────────┬────────────────────────────────────┘
                        │  POST /predict_route (JSON)
                        ▼
┌────────────────────────────────────────────────────────────┐
│           UK-07 FastAPI Backend  (app.py)                  │
│   Deployed: HuggingFace Spaces (CPU Docker)               │
│                                                            │
│  1. Hybrid Geocoding   → TomTom Search API                │
│  2. GNN Inference      → ElevationAwareGAT (GATv2)        │
│  3. Dijkstra Routing   → NetworkX shortest_path()         │
└────────────────────────────────────────────────────────────┘
                        │  Uses
                        ▼
┌────────────────────────────────────────────────────────────┐
│              Trained Model & Graph Data (HF LFS)           │
│   Models/traffic_model_v1.pth                             │
│   Dataset/dehradun_mussoorie_full.graphml                  │
│   Dataset/dm_graph_tensors.pt                             │
└────────────────────────────────────────────────────────────┘
```

---

## 📁 Repository Structure

```
DeepLearn/
│
├── Project/                    # ML pipeline scripts (run in order)
│   ├── project_Step_1.py       # Download OSM road network (osmnx)
│   ├── project_Step_2.py       # Feature engineering + elevation (SRTM)
│   ├── project_Step_3.py       # Fetch live TomTom traffic labels
│   ├── project_Step_4.py       # GATv2 model definition & training
│   ├── test.py                 # Quick API sanity-check script
│   └── app.py                  # ⚡ NOT here — lives on HuggingFace Spaces
│                               #    → https://huggingface.co/spaces/Sarthak2606/UK-07_Smart_Traffic_Engine
│
├── TerrainNav_App/             # PWA frontend (Vanilla JS)
│   ├── index.html
│   ├── app.js                  # GPS capture, UI logic
│   ├── map.js                  # Leaflet map rendering, route drawing
│   ├── api.js                  # Backend API calls
│   ├── config.js               # ⚠️ Fill in YOUR TomTom keys here
│   ├── styles.css
│   ├── manifest.json           # PWA manifest
│   └── sw.js                   # Service worker (offline cache)
│
├── TerrainNav_APK/             # Capacitor Android build wrapper
│   ├── capacitor.config.json
│   └── package.json
│
├── DeepLearn_Documentation.md  # Deep technical walkthrough
├── requirements.txt            # Python dependencies
├── .env.example                # ← Copy to .env and add your keys
└── .gitignore
```

> **Dataset/** and **Models/** directories are excluded from Git (large binary files).  
> They are stored on [HuggingFace Spaces LFS](https://huggingface.co/spaces/Sarthak2606/UK-07_Smart_Traffic_Engine/tree/main).

---

## 🚀 Quick Start

### 1. Clone & set up environment

```bash
git clone https://github.com/<your-username>/DeepLearn.git
cd DeepLearn
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure API keys

```bash
cp .env.example .env
# Edit .env and add your TomTom API keys:
# TOMTOM_KEY_MORNING=...
# TOMTOM_KEY_AFTERNOON=...
# TOMTOM_KEY_EVENING=...
```

Get free TomTom keys at → **https://developer.tomtom.com/**

### 3. Run the ML Pipeline (training)

> Skip this if you just want to run the API — the pre-trained model is on HuggingFace.

```bash
cd Project
python project_Step_1.py    # Download OSM graph  (~2 min)
python project_Step_2.py    # Enrich with elevation & features (~10 min)
python project_Step_3.py    # Fetch live traffic labels (uses API quota)
python project_Step_4.py    # Train the GATv2 model (~varies)
```

### 4. Run the backend locally

```bash
cd Project
uvicorn app:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Run the frontend

Open `TerrainNav_App/index.html` directly in a browser, or serve it:

```bash
cd TerrainNav_App
npx serve .
```

> ⚠️ **Edit `config.js`** before running — add your TomTom API keys and optionally change `BACKEND_URL` to your local server (`http://localhost:8000`).

---

## 🌐 Live Deployment

| Component | URL |
|-----------|-----|
| **FastAPI Backend** | https://sarthak2606-uk-07-smart-traffic-engine.hf.space |
| **API Docs (Swagger)** | https://sarthak2606-uk-07-smart-traffic-engine.hf.space/docs |
| **HuggingFace Space** | https://huggingface.co/spaces/Sarthak2606/UK-07_Smart_Traffic_Engine |

### Example API call

```bash
curl -X POST https://sarthak2606-uk-07-smart-traffic-engine.hf.space/predict_route \
  -H "Content-Type: application/json" \
  -d '{"start_name": "Clock Tower", "end_name": "Mussoorie Mall Road"}'
```

```json
{
  "status": "success",
  "path": [[30.324, 78.042], ...],
  "eta_mins": 28.5,
  "prediction_time_ist": "2026-04-26_17-30"
}
```

---

## 🤖 Model: ElevationAwareGAT

The core model is a **Graph Attention Network v2 (GATv2)** with residual connections:

| Feature | Detail |
|---------|--------|
| Architecture | GATv2Conv × 2 layers + MLP regressor |
| Node features | Latitude, Longitude, Degree, Elevation |
| Edge features | Length, Speed, Travel time, One-way, Lanes, Grade, Highway type (8-class OHE) |
| Temporal input | Sin/cos encodings of hour-of-day + day-of-week |
| Output | Traffic factor per edge (0.0–1.0, where 1.0 = free flow) |
| Loss function | Custom Weighted MSE (heavy congestion penalized 120×) |
| Map coverage | Dehradun–Mussoorie corridor (~45 km radius) |

See **[DeepLearn_Documentation.md](./DeepLearn_Documentation.md)** for the full technical deep-dive.

---

## 🔑 Environment Variables

| Variable | Description |
|----------|-------------|
| `TOMTOM_KEY_MORNING` | TomTom API key used 06:00–11:59 IST |
| `TOMTOM_KEY_AFTERNOON` | TomTom API key used 12:00–17:59 IST |
| `TOMTOM_KEY_EVENING` | TomTom API key used 18:00–05:59 IST |

Copy `.env.example` → `.env` and fill in your keys. Never commit `.env`.

---

## 🛡️ Security Note

**API keys are never hardcoded in this repository.**  
- Python backend reads keys via `python-dotenv` from `.env`  
- Frontend `config.js` uses placeholder strings — replace them locally  
- `.env` is listed in `.gitignore`

---

## 📄 License

This project is for educational and demonstration purposes.  
Built with ❤️ for **Uttarakhand** 🏔️
