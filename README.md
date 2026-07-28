# Brain Tumor Identification System

**Published at ICCIES 2025** (International Conference on Computational Intelligence in Engineering Science)

---

## Overview

This repository contains training and inference code for binary classification of brain tumors from MRI scans (Tumor / No Tumor). The pipeline combines **MobileNet** feature extraction with **PCA** dimensionality reduction, then applies a decision-level fuzzy ensemble using **Gompertz** or **Mitscherlich** functions across multi-scale 1D-CNNs, with **SVM** as an additional baseline classifier.

📄 **[Full Technical Documentation](docs/TECHNICAL_REPORT.md)** — system architecture, inference pipeline, API reference, and project scope.

> ⚠️ Research prototype for medical-AI research and educational use; **not intended for clinical use.** Input images must be single-slice axial, skull-stripped MRI scans.

---

## Repository Structure

```text
gompertz-iccies.ipynb      # Training and evaluation pipeline (Gompertz ensemble)
mitscherlich-iccies.ipynb  # Training and evaluation pipeline (Mitscherlich ensemble)
svm-iccies.ipynb           # Training and evaluation pipeline (SVM baseline)
inference.py               # Inference module (file path, URL, bytes, or PIL Image)
main.py                    # FastAPI server (REST API)
app.py                     # Streamlit demo UI
preprocessing.py           # Brain region isolation via OpenCV
requirements.txt           # Full dependencies (single machine)
requirements-api.txt       # API-only dependencies (Docker)
requirements-ui.txt        # UI-only dependencies (Docker)
Dockerfile.api             # Container image for the FastAPI service
Dockerfile.ui              # Container image for the Streamlit UI
docker-compose.yml         # Orchestrates the API + UI services
docs/
    TECHNICAL_REPORT.md    # Full technical documentation
models/                    # Model weights (PCA downloaded separately — see Setup)
    Gompertz/
    Mitscherlich/
    SVM/
```

---

## Setup: Download PCA Weights

The CNN and SVM weights are included in this repository. The **PCA weights** (`pca_model.joblib`, 218 MB) are hosted separately. Download the file and place a copy in **each** of the three model folders.

📦 **[Download pca_model.joblib (Google Drive)](https://drive.google.com/drive/folders/1Q_Z8kDkTndmOCi0Cu7eZ9sXJixwPuOSK)**

```bash
# place the same file into all three folders:
cp pca_model.joblib models/Gompertz/pca_model.joblib
cp pca_model.joblib models/Mitscherlich/pca_model.joblib
cp pca_model.joblib models/SVM/pca_model.joblib
```

After setup, each folder under `models/` should contain its `.keras`/`.joblib` weights plus `pca_model.joblib`.

> ℹ️ **First run requires internet.** On the first inference, the MobileNet backbone downloads its ImageNet weights from PyTorch Hub (cached afterwards).

---

## Datasets

- **Raw MRI scans**: [Kaggle](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor) — licensed under CC BY-NC-SA 4.0
- **Preprocessed (brain region isolated)**: [Kaggle](https://www.kaggle.com/datasets/sealeopard/reloaded-crop) — derived from above, licensed under CC BY-NC-SA 4.0

---

## Running with Docker (recommended)

Two containerised services — a **FastAPI inference API** and a **Streamlit UI** — orchestrated via Docker Compose.

```bash
docker compose up --build
```

- API → `http://localhost:8000`
- UI  → `http://localhost:8501`

PCA weights are mounted from `models/` at runtime, so complete the *Setup* step above before starting the containers.

> First build takes a while (PyTorch + TensorFlow are large). Subsequent builds are cached.

### AWS Lambda deployment (optional)

The API also supports serverless deployment on AWS Lambda. In `Dockerfile.api`, comment out the default `uvicorn` CMD and enable the `awslambdaric` line, then point the UI's *API Base URL* (sidebar) at your Lambda function URL.

---

## Running Without Docker

```bash
pip install -r requirements.txt
```

### API Server

```bash
export MODEL_ROOT=./models        # Windows: set MODEL_ROOT=.\models
uvicorn main:app --host 0.0.0.0 --port 8000
```

`MODEL_ROOT` must point to your local `models/` directory. API endpoints:
- `POST /predict/upload` — classify from uploaded image file
- `POST /predict/url` — classify from image URL
- `POST /predict` — combined endpoint (file or URL)
- `GET /health` — server health check

Supported methods: `Gompertz`, `Mitscherlich`, `SVM`

### Demo UI

```bash
export API_BASE_URL=http://localhost:8000
streamlit run app.py
```

Open the URL Streamlit provides. The API base URL can also be changed in the sidebar.

---

## Running the Training Notebooks

All training notebooks are designed to run on **Kaggle** (free GPU):
1. Upload the dataset to Kaggle.
2. Open the desired notebook.
3. Update dataset paths if necessary.
4. Run all cells.

---

## Preprocessing

Raw MRI scans are brain-region isolated before use. Inference applies this automatically; to preprocess a folder manually:

```bash
python preprocessing.py <input_folder> <output_folder>
```