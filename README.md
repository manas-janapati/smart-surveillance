# Smart Surveillance of Harmful Highway Vegetation Using Deep Learning

## Project Overview
This project focuses on building an **AI-based smart surveillance system** for detecting **harmful invasive vegetation along highways** using **deep learning and computer vision**.

The long-term goal is to support **automated roadside monitoring**, enabling timely identification of invasive plant growth and assisting environmental and highway authorities in taking preventive action.

---

## Current Milestone (Successful Implementation)

We have successfully completed the **first functional implementation** of the project, which validates the core deep learning inference pipeline.

### What Has Been Achieved So Far

- Integrated **pre-trained deep learning models** provided from prior research
- Successfully loaded and combined:
  - **BEiT backbone model** (feature extractor)
  - **XP5 classifier head** trained on invasive plant data
- Implemented a **CPU-based inference pipeline**
- Achieved reliable **binary classification**:
  - **Invasive vegetation**
  - **Non-invasive / background vegetation**
- No retraining or fine-tuning was performed (inference-only deployment)

---

## Models Used

- **BEiT Backbone**
  - Pre-trained on plant datasets
- **XP5 Classifier Head**
  - Trained on invasive roadside vegetation
  - Outputs confidence scores for invasive plant patterns

The system uses **confidence thresholding** on model outputs to determine whether vegetation is invasive or non-invasive.

---

## Current System Capabilities

✔ Image preprocessing aligned with training configuration  
✔ Model loading with correct architecture reconstruction  
✔ Inference on sample images  
✔ Robust invasive vs non-invasive decision logic  
✔ Runs fully on CPU (edge-friendly)  

