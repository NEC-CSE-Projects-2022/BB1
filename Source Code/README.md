# BB1 — Frontend Overview

A clear, professional README for the frontend of the **BB1** project. This document describes the project structure, how to run the app locally, the **AI models used**, and the **input image validation** performed before analysis.

## Table of contents

- Project Structure
- How to run (Frontend)
- Folder & file explanations
- Models used
- Input validation (Skin lesion image validation)
- Frontend Screenshots
- Notes

## Project Structure

- **Source Code**
  - **app.py** — Flask server entrypoint (routes, inference, validation, DB writes)
  - **requirements.txt** — Python dependencies

- **Templates & UI**
  - **templates/** — HTML templates (welcome/login/signup/dashboard, etc.)
  - **static/** — CSS/JS assets and uploads

- **Models & Data**
  - **models/** — saved model weights/artifacts used for inference

- **Screenshots**
  - **images/** — frontend screenshots (displayed below)

## How to run (Frontend)

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start the app

```bash
python app.py
```

Open the address printed by Flask (commonly `http://127.0.0.1:5000`).

## Folder & file explanations

- **templates/**
  - Contains the UI pages rendered by Flask.

- **static/**
  - Contains styling and client-side behavior.
  - `static/uploads/` is used for uploaded images during analysis.

- **models/**
  - Contains the trained model files referenced by the backend.

- **images/**
  - Screenshots used in this README.

## Models used

The backend uses an ensemble-style setup with the following components:

- **EfficientFormer (via `timm`)**
  - Used as one of the image classification backbones.

- **Swin Transformer (via `timm`)**
  - Used as a second image classification backbone.

- **Meta-learner (Logistic Regression)**
  - A lightweight model that combines the outputs from the backbones to produce the final prediction.

The system predicts **7 skin lesion classes**.

## Input validation (Skin lesion image validation)

Before running inference, the app validates whether the uploaded image looks like a valid skin lesion image. The validation is designed to reduce incorrect analysis on non-medical or unrelated images.

Validation includes:

- **Image dimension checks** (minimum size requirements)
- **Dominant color analysis** using clustering
- **Skin-like color ratio checks** (ensures the image contains sufficient skin-tone regions)
- **Unnatural color detection** (filters obvious landscapes/outdoor scenes)
- **Contrast/variance checks** (rejects overly-uniform images)

If validation fails, the API returns a user-friendly error message and the image is not analyzed.

## Frontend Screenshots

Below are the screenshots currently in `images/`. Each image is shown as a preview and links to the full file.

| 1                                                                                                                                                           | 2                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <a href="images/1.png"><img width="1897" height="1079" alt="1" src="https://github.com/user-attachments/assets/7056c36d-b9be-4122-86ea-9c56bae2fa5e" /></a> | <a href="images/2.png"><img width="1893" height="1069" alt="2" src="https://github.com/user-attachments/assets/41c7ab4e-19e6-465c-807a-3af6bbd31bb0" /></a> |

| 3                                                                                                                                                           | 4                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <a href="images/3.png"><img width="1898" height="1079" alt="3" src="https://github.com/user-attachments/assets/b35d8917-ff77-47ea-a211-ef4a20c618af" /></a> | <a href="images/4.png"><img width="1919" height="1079" alt="4" src="https://github.com/user-attachments/assets/896bc255-5a1e-4844-9aaf-192daa158185" /></a> |

| 5                                                                                                                                                           | 6                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <a href="images/5.png"><img width="1894" height="1079" alt="5" src="https://github.com/user-attachments/assets/2490edbb-3a9f-46e2-be36-919da2328c5f" /></a> | <a href="images/6.png"><img width="1903" height="1079" alt="6" src="https://github.com/user-attachments/assets/9cee213f-872a-41cb-9583-500168bb0ebf" /></a> |

| 7                                                                                                                                                           | 8                                                                                                                                                           |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| <a href="images/7.png"><img width="1900" height="1079" alt="7" src="https://github.com/user-attachments/assets/4928cf74-c32a-49e7-b753-2028e33db97f" /></a> | <a href="images/8.png"><img width="1919" height="1079" alt="8" src="https://github.com/user-attachments/assets/5b03ea4b-7832-4239-b097-b3cbda3b5c28" /></a> |

Captions:

- **1** — Welcome page
- **2** — Login page
- **3** — Sign up page
- **4** — Dashboard (normal / initial view)
- **5** — Dashboard (image uploaded)
- **6** — Detecting lesions (analysis in progress / result view)
- **7** — Validation (non-skin or invalid image handling)
- **8** — Thank you page
