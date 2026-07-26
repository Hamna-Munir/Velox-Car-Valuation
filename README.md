<div align="center">

# 🏁 VELOX
### Vehicle Valuation Engine

A production-structured machine learning system that predicts **used car resale value**
from real-world market data — quantile regression for genuine uncertainty bands,
SHAP explainability, a FastAPI service, and a Streamlit front end, all sharing one
tested core library.

[![Python](https://img.shields.io/badge/Python-3.10+-0A0A0A?style=for-the-badge&logo=python&logoColor=E10600)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-0A0A0A?style=for-the-badge&logo=streamlit&logoColor=E10600)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0A0A0A?style=for-the-badge&logo=fastapi&logoColor=E10600)](https://fastapi.tiangolo.com)
[![scikit--learn](https://img.shields.io/badge/scikit--learn-0A0A0A?style=for-the-badge&logo=scikitlearn&logoColor=E10600)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/License-MIT-0A0A0A?style=for-the-badge&color=E10600)](#license)

[Live Demo](#) · [Features](#features) · [Architecture](#architecture) · [Getting Started](#getting-started)

</div>

---

## Overview

VELOX estimates resale value for used cars — Maruti, Hyundai, Honda, Toyota,
and more — based on brand, age, kilometers driven, fuel type, transmission,
seller type, and ownership history.

What separates this from a "train one model, wrap it in a form" project:

- **Quantile regression**, not a heuristic error margin — the P10/P50/P90
  range shown for every prediction is three separately-trained models, so
  it's a real estimate of *this vehicle's* uncertainty, not just the
  historical spread of its brand.
- **SHAP explainability** — every prediction ships with a per-field
  breakdown of what pushed the price up or down, not just an aggregate
  "feature importance" chart.
- **Hyperparameter search with cross-validation** (`RandomizedSearchCV`,
  5-fold) — the reported R² comes from data the search never touched.
- **A tested, importable core library** (`velox/`) shared by both the
  Streamlit app and a separate FastAPI service, with 22 passing tests
  covering data validation, model behavior, and the API contract.
- **CI, Docker, and environment-based config** — the parts of a real
  deployment pipeline that don't show up in a demo screenshot but matter
  the moment more than one person touches the project.

<table>
<tr>
<td width="50%" valign="top">

**What it does**
- Predicts resale value with a calibrated P10–P90 range
- Explains *why*, per-prediction, via SHAP
- Projects a full depreciation curve across vehicle age
- Exports a branded PDF valuation report
- Serves the same model over a REST API

</td>
<td width="50%" valign="top">

**What makes it engineering-grade, not just a demo**
- `velox/` is a real Python package: config, data, model, explainability,
  schemas — each independently testable
- 22 pytest tests, run in CI on every push
- FastAPI service with Pydantic-validated requests (422 on bad input,
  not a stack trace)
- Dockerized (app + API as separate services)
- No hardcoded paths — everything is environment-configurable

</td>
</tr>
</table>

---

## Architecture

```
                          ┌─────────────────────────────┐
                          │   data/car_details.csv        │
                          │   (raw Kaggle export)         │
                          └───────────────┬─────────────┘
                                          │
                                          ▼
                          ┌─────────────────────────────┐
                          │   velox/  (core library)      │
                          │                               │
                          │   config.py   — env-driven     │
                          │                 settings       │
                          │   data.py     — clean + validate│
                          │   model.py    — tune, train,   │
                          │                 quantile models│
                          │   explain.py  — SHAP, global   │
                          │                 importance     │
                          │   schemas.py  — Pydantic I/O    │
                          └───────┬───────────────┬───────┘
                                  │               │
                     ┌────────────┘               └────────────┐
                     ▼                                          ▼
        ┌─────────────────────────┐                ┌─────────────────────────┐
        │   train_model.py          │                │   tests/                  │
        │   (CLI entry point)       │                │   test_data / test_model /│
        │                           │                │   test_api — 22 tests,    │
        │   → model.pkl              │                │   run in CI on every push │
        │   → quantile_models.pkl    │                └─────────────────────────┘
        │   → metrics.pkl             │
        └────────────┬──────────────┘
                     │
        ┌────────────┴─────────────┐
        ▼                           ▼
┌─────────────────┐      ┌──────────────────────┐
│   app.py           │      │   api/main.py          │
│   Streamlit UI      │      │   FastAPI service      │
│   (uses velox/       │      │   /predict /health      │
│   directly, no       │      │   /brands               │
│   network hop)        │      │   (independent consumer)│
└─────────────────┘      └──────────────────────┘
```

**Why two front ends?** `app.py` imports `velox` directly — fast,
single-process, ideal for Streamlit Cloud, which only runs one container.
`api/main.py` exposes the *same* underlying models over HTTP for any other
consumer (a mobile app, another backend, a batch pricing job). Both share
one core library, so there's exactly one place the modeling logic lives —
change it once, both surfaces pick it up.

---

## Features

### Chapter 01 — Configure
- Vehicle profile form — brand, manufacturing year, kilometers driven, fuel
  type, transmission, seller type, ownership
- Live resale value estimate with a **quantile-regression-based** confidence
  range (P10/P50/P90, predicted for this exact configuration)
- Market band gauge visualizing where the estimate sits in that range
- **"Why This Estimate"** — a live SHAP waterfall showing each field's
  contribution to this specific prediction
- Comparison chips against brand / fuel-type / overall medians
- **PDF report export** — one-click branded, print-ready valuation report

### Chapter 02 — The Depreciation Story
- Predicted value at every age from 0–25 years for the current vehicle
  configuration, with 10-year value-loss %

### Chapter 03 — Showroom Gallery
- Illustrated body-style gallery (sedan / SUV / hatchback / coupe) — original
  gradient-shaded SVG illustrations, not stock photos (see [Design
  System](#design-system) for why)
- Price by brand, by fuel type, by vehicle age, vs kilometers driven

### Chapter 04 — Under the Hood
- Data preparation notes, including the 2026 market-price adjustment
- Model performance: test R², 5-fold CV R², MAE, RMSE, and P10–P90 coverage
  (the share of true prices that actually fall inside the predicted range —
  ~80% means the uncertainty band is well-calibrated, not just wide)
- **Global** feature importance, distinct from the **per-prediction** SHAP
  breakdown shown on the Predict chapter

---

## Dataset

[**Car Details from Car Dekho**](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
— Nehal Birla, Kaggle. 4,340 real listings, India, 1992–2020 (manufacturing years).

Bundled at `data/car_details.csv` so training works fully offline — no
Kaggle API key required.

**Pricing is scaled to 2026.** The raw listing prices reflect the Indian
used-car market as of ~2021 — genuinely fresh 2026 listings aren't available
as an open dataset. Rather than show stale 2021 prices, `velox/data.py`
scales every price up using a documented **~8%/year** used-car price
appreciation rate for India (Cars24/Team-BHP and Mordor Intelligence market
reports, 2024–2025), landing on 2026-equivalent market levels. This is a
transparent, disclosed **estimate** — the app says so explicitly in its
"A Note on 2026 Pricing" card — not real 2026 transaction data. Change
`VELOX_ANNUAL_APPRECIATION` / `VELOX_CURRENT_YEAR` (see `.env.example`) to
adjust the assumption or roll it forward in future years.

| Field | Description |
|---|---|
| `brand` | Extracted from listing name, reduced to top 12 + "Other" |
| `car_age` | Derived from manufacturing year (relative to 2026) |
| `km_driven` | Odometer reading at listing time |
| `fuel` | Petrol / Diesel / CNG / LPG / Electric |
| `seller_type` | Individual / Dealer / Trustmark Dealer |
| `transmission` | Manual / Automatic |
| `owner` | First Owner → Fourth & Above / Test Drive Car |
| `price_inr` | Target variable — listing price in Indian Rupees, scaled to 2026 |

**A note on real data.** Brand, age, mileage, and fuel type explain most of
the variance in used-car prices — but this dataset doesn't capture a
vehicle's actual condition, accident history, service records, or
negotiation. The P10–P90 band communicates that uncertainty explicitly
instead of hiding behind a single confident-looking number.

---

## Design System

VELOX uses a black / white / red racing-inspired aesthetic, laid out as a
**single-scroll "showroom" experience** rather than a tabbed dashboard —
a full-bleed hero, then four numbered chapters you scroll through in sequence.

Every car graphic is an **original, hand-built gradient-shaded SVG** (body,
cabin glass, alloy wheels, lights), not a stock photo. That was a deliberate
choice: hotlinked third-party photos are fragile (many image hosts,
including Wikimedia, block hotlinking from cloud/datacenter IPs — exactly
where Streamlit Cloud runs) and carry licensing obligations. An original
illustration has neither problem and renders identically anywhere.

<div align="center">

| Token | Swatch | Hex |
|---|---|---|
| Background | ![#0A0A0A](https://placehold.co/60x20/0A0A0A/0A0A0A.png) | `#0A0A0A` |
| Panel | ![#141414](https://placehold.co/60x20/141414/141414.png) | `#141414` |
| Border | ![#2A2A2A](https://placehold.co/60x20/2A2A2A/2A2A2A.png) | `#2A2A2A` |
| Text | ![#FFFFFF](https://placehold.co/60x20/FFFFFF/FFFFFF.png) | `#FFFFFF` |
| Muted text | ![#9A9A9A](https://placehold.co/60x20/9A9A9A/9A9A9A.png) | `#9A9A9A` |
| Racing red | ![#E10600](https://placehold.co/60x20/E10600/E10600.png) | `#E10600` |
| Red bright | ![#FF2A2A](https://placehold.co/60x20/FF2A2A/FF2A2A.png) | `#FF2A2A` |
| Chrome | ![#F2F2F2](https://placehold.co/60x20/F2F2F2/F2F2F2.png) → ![#5C5C5C](https://placehold.co/60x20/5C5C5C/5C5C5C.png) | `#F2F2F2` → `#5C5C5C` |

</div>

**Typography** — Oswald (condensed display headings) · Inter (body) · IBM Plex Mono (all numeric data)

---

## Tech Stack

| Layer | Tools |
|---|---|
| **Modeling** | scikit-learn `GradientBoostingRegressor` (point estimate + 3 quantile models), `RandomizedSearchCV` with 5-fold CV |
| **Explainability** | SHAP `TreeExplainer` (per-prediction) + aggregated `feature_importances_` (global) |
| **API** | FastAPI, Pydantic v2 request/response validation, CORS-enabled |
| **App / UI** | Streamlit (`segmented_control`, bordered containers) |
| **Visualization** | Plotly (bar, box, line, histogram, scatter, SHAP waterfall) |
| **Reporting** | ReportLab (PDF generation) |
| **Testing** | pytest, FastAPI `TestClient`, 22 tests |
| **CI/CD** | GitHub Actions (lint + test on every push) |
| **Packaging** | `pyproject.toml`, Docker (app + API as separate images), `docker-compose` |
| **Data** | Pandas, NumPy |

---

## Getting Started

### Prerequisites
- Python 3.10+
- ~1.5 GB free disk space (scikit-learn, SHAP, and FastAPI add up)

### Installation

```bash
git clone <your-repo-url>
cd velox

python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### Train the models

```bash
python train_model.py
```

This runs the hyperparameter search (25 iterations × 5-fold CV, ~1–2
minutes), trains the 3 quantile models, and saves `model.pkl`,
`quantile_models.pkl`, `metrics.pkl`, and `car_data_clean.csv`. The app and
API will also auto-train on first load if these are missing — running it
explicitly first just makes that first load instant.

### Run the Streamlit app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (usually `http://localhost:8501`).

### Run the API (optional, separate service)

```bash
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

Interactive docs at `http://localhost:8000/docs`. Example request:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "brand": "Maruti", "year": 2021, "km_driven": 40000,
        "fuel": "Petrol", "transmission": "Manual",
        "seller_type": "Individual", "owner": "First Owner"
      }'
```

### Run tests

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

### Run with Docker

```bash
docker compose up --build
```

Starts the Streamlit app on `:8501` and the API on `:8000` as separate
containers, each training its own models at build time.

### Troubleshooting

<details>
<summary><b>ModuleNotFoundError after install</b></summary>

Usually means `pip install -r requirements.txt` didn't finish. A common cause:

```
ERROR: Could not install packages due to an OSError: [Errno 28] No space left on device
```

Free up disk space (check both the drive your venv is on **and** `C:` —
pip's temp extraction directory defaults to `C:` on Windows regardless of
where your venv lives), then:

```bash
pip cache purge
pip install -r requirements.txt --no-cache-dir
```

</details>

<details>
<summary><b>streamlit / python -m streamlit says "No module named streamlit"</b></summary>

Your virtual environment isn't actually active, or a global Python
install is shadowing it. Confirm with:

```bash
where python
where streamlit
```

Both should point inside your project's `venv\Scripts\`. If not:

```bash
deactivate
venv\Scripts\activate
pip install -r requirements.txt --no-cache-dir
```

</details>

<details>
<summary><b>ValueError: ... is not a known BitGenerator module</b></summary>

A `model.pkl` pickled with a different numpy version than the one installed
locally. Delete the generated artifacts and let the app retrain with your
local versions:

```bash
del model.pkl quantile_models.pkl metrics.pkl car_data_clean.csv      # Windows
# rm model.pkl quantile_models.pkl metrics.pkl car_data_clean.csv    # macOS / Linux
streamlit run app.py
```

</details>

---

## Project Structure

```
velox/
├── velox/                      # core library (importable, tested)
│   ├── __init__.py
│   ├── config.py                # env-driven settings, paths, constants
│   ├── data.py                  # load/clean/validate the dataset
│   ├── model.py                 # hyperparameter search, quantile models
│   ├── explain.py               # SHAP + global feature importance
│   ├── schemas.py               # Pydantic request/response models
│   └── logging_config.py
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI service (/predict /health /brands)
├── tests/
│   ├── test_data.py              # 8 tests
│   ├── test_model.py             # 7 tests
│   └── test_api.py               # 7 tests
├── data/
│   └── car_details.csv          # bundled raw Kaggle dataset
├── .github/workflows/ci.yml     # lint + test on every push
├── app.py                        # Streamlit app (imports velox/ directly)
├── train_model.py                # thin CLI wrapper around velox.model
├── Dockerfile                    # Streamlit app image
├── Dockerfile.api                # FastAPI service image
├── docker-compose.yml
├── pyproject.toml                # packaging + pytest/ruff config
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .streamlit/config.toml        # dark theme base
├── .gitignore
├── LICENSE
└── README.md
```

---

## Deployment

### Streamlit Community Cloud (the app)

1. Push this repo to GitHub (including `data/car_details.csv`).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select the repo and branch, set the main file to `app.py`.
4. Deploy — dependencies install from `requirements.txt` and the model
   trains automatically on first load.

### The API (any container host)

`Dockerfile.api` builds a standalone image exposing port 8000. Deploy it
anywhere that runs containers (Render, Railway, Fly.io, AWS ECS, etc.) —
it has no dependency on Streamlit or the app.

### Refreshing the dataset

Swap in a newer export from the same or a similar source (matching columns:
`name`, `year`, `selling_price`, `km_driven`, `fuel`, `seller_type`,
`transmission`, `owner`), replace `data/car_details.csv`, delete any cached
`model.pkl` / `quantile_models.pkl`, and rerun `python train_model.py`.

---

## Model Performance

Held-out 20% test split, point-estimate model tuned via `RandomizedSearchCV`:

| Metric | Value |
|---|---|
| Test R² | ~0.81 |
| 5-fold CV R² | ~0.77 |
| MAE | ~Rs 199,000 |
| RMSE | ~Rs 344,000 |
| P10–P90 coverage | ~78% (target ~80%) |

The P10–P90 coverage figure is the share of true held-out prices that
actually land inside the model's predicted range — the calibration check
that separates a real uncertainty estimate from a cosmetic one.

---

## Roadmap

- [ ] Vehicle condition / accident-history as a feature
- [ ] City/region-level pricing (location-adjusted estimates)
- [ ] Model registry (MLflow) instead of flat `.pkl` files
- [ ] Scheduled retraining via GitHub Actions cron
- [ ] Auth + rate limiting on the API for public deployment

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

## Author

**Hamna Munir**
Software Engineering & AI/ML

<p>
<a href="#"><img src="https://img.shields.io/badge/GitHub-0A0A0A?style=for-the-badge&logo=github&logoColor=E10600" /></a>
<a href="#"><img src="https://img.shields.io/badge/LinkedIn-0A0A0A?style=for-the-badge&logo=linkedin&logoColor=E10600" /></a>
</p>

<sub>Update the badge links above with your actual profile URLs.</sub>

---

<div align="center">
<sub>Built as a portfolio ML engineering project · Dataset © respective Kaggle contributor</sub>
</div>
