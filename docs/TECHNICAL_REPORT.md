# Brain Tumor MRI Classifier

**AI-Assisted Diagnostic Support System**
Project Report | 2026

---

## 1. Project Summary

The Brain Tumor MRI Classifier is a research demo system that classifies brain MRI scans as either tumor-present or tumor-absent using a multi-model ensemble approach. The system provides a web interface for researchers to submit MRI images — either by direct upload or via URL — and inspect classification outputs, without requiring direct access to the underlying codebase or model weights.

### Goal

To provide an interactive demo environment for evaluating a multi-model ensemble approach to binary brain MRI classification — enabling researchers to submit axial MRI images and inspect classification outputs.

This system is not intended for clinical deployment. It serves as a demonstration interface for the underlying research pipeline.

| | |
|---|---|
| **Project Name** | Brain Tumor MRI Classifier |
| **Domain** | Medical AI / Radiology Assistance |
| **Platform** | Web Application (FastAPI + Streamlit) |
| **Deployment** | AWS Lambda (API) + EC2 (UI) |
| **Models Used** | MobileNet, Custom Shallow CNN, SVM with Fuzzy Ensemble |
| **Training Dataset** | [Kaggle — Brain Tumor](https://www.kaggle.com/datasets/jakeshbohaju/brain-tumor) |

---

## 2. Project Statement and Scope

### Problem Statement

This system was developed to support the evaluation phase of a brain tumor MRI classification research project, providing a lightweight web interface that allows stakeholders to submit axial MRI images and inspect ensemble model outputs without requiring direct access to the codebase.

### Scope

This project covers the development and deployment of a research demo web application for brain tumor binary classification, within the following scope boundaries:

**In scope:**
- MRI image classification (upload and URL input), multi-model ensemble scoring, REST API, Streamlit web UI, cloud deployment on AWS
- Binary classification — tumor present vs. no tumor detected
- Input images must be single-slice axial (horizontal) plane MRI scans with the skull stripped. Coronal and sagittal orientations are not supported.

**Out of scope:**
- DICOM file ingestion pipeline, integration with hospital PACS systems, real-time streaming inference, model retraining workflows
- Regulatory approval or clinical certification (system is intended for research and decision-support use only)

---

## 3. Technical Description

### System Architecture

The system is structured as a two-tier web application: a FastAPI backend that handles inference, and a Streamlit frontend that provides the user interface. Both services are containerised using Docker and deployed on AWS.

| Component | Technology | Role |
|-----------|------------|------|
| Inference Engine | PyTorch (MobileNet), TensorFlow (CNN), Scikit-learn (SVM) | Extracts image features via MobileNet and runs inference across three CNNs and one SVM classifier |
| Ensemble Logic | Fuzzy logic (Gompertz & Mitscherlich) | Combines model outputs into a final confidence-weighted prediction |
| Backend API | FastAPI + Uvicorn | Handles image upload and URL-based prediction requests |
| Frontend UI | Streamlit | Web interface for submitting MRI images and reviewing classification results during research evaluation sessions |
| Cloud Infra | AWS Lambda + EC2 | Serverless API with pay-per-request; EC2 hosts UI |

### Inference Pipeline

Classification follows a two-stage pipeline. In the first stage, a pretrained MobileNet backbone (PyTorch) processes the input MRI image as a fixed feature extractor with no fine-tuning applied, producing a feature vector consumed by all downstream classifiers.

In the second stage, four classifiers operate independently on the extracted features:

- **CNN (TensorFlow/Keras):** Three custom shallow 1D CNNs, each trained independently on MobileNet-extracted features.
- **SVM (Scikit-learn):** A support vector machine trained on the same features, serving as a non-neural baseline.

The three CNN outputs are aggregated using fuzzy ensemble logic implementing Gompertz and Mitscherlich functions to weight and combine confidence scores into a final prediction. The SVM produces its prediction independently. The user-selected method (SVM, Gompertz, or Mitscherlich) determines which prediction path is returned.

### API Endpoints

- `POST /predict/upload` — Accepts multipart image file upload, returns classification result
- `POST /predict/url` — Accepts a publicly accessible image URL, fetches and classifies
- `POST /predict` — General prediction endpoint
- `GET /health` — Returns server status and model load confirmation

### Deployment

The API is deployed as a containerised AWS Lambda function, providing pay-per-request serverless execution with no idle infrastructure cost. The Streamlit UI is hosted on an EC2 t3.micro instance, started on demand for demo sessions and stopped when not in use.

Public access is provided via Cloudflare Tunnel, enabling HTTPS without requiring a custom domain or SSL certificate configuration.

---

## 4. User Personas

The following personas reflect the research-oriented context of this system. The primary users are researchers and evaluators involved in the development or review of brain MRI classification models, not clinical practitioners.

| Theme | Who (Position) | I Can... |
|-------|----------------|----------|
| Pipeline Evaluation | Medical AI researcher / graduate student | Submit axial brain MRI images via upload or URL to inspect the ensemble model's binary classification output. |
| Model Benchmarking | Research supervisor / paper reviewer | Use the web interface to demonstrate the end-to-end classification pipeline interactively, without requiring access to the underlying codebase or model weights. |

> **Note:** This system is not intended for use with real patient data or in any clinical setting. All interactions are assumed to take place within a controlled research environment.