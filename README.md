# 🔥 Calorie Prediction via Gradient Boosted Ensemble (Kaggle S5E5)

This repository contains a complete solution for the [Kaggle Playground Series - Season 5, Episode 5](https://www.kaggle.com/competitions/playground-series-s5e5) competition. The objective is to predict calories burned based on biometric and physiological features.

---

## 📌 Overview

- **Problem Type:** Regression  
- **Target Variable:** `Calories`  
- **Evaluation Metric:** RMSLE (Root Mean Squared Log Error)  
- **Dataset Size:** 750,000 training samples, 250,000 test samples

---

## ⚙️ Approach Summary

- **Log Target Transformation:** `log1p(Calories)` to match RMSLE behavior
- **Models Used:**
  - LightGBM (GPU)
  - XGBoost (GPU histogram)
  - CatBoost (GPU)
- **Feature Engineering:**
  - BMI (Body Mass Index)
  - MET Proxy (`Duration × HR × Temp`)
  - Interaction features: `HR × Duration`, `Weight × Duration`, `Age × HR`
- **Categorical Encoding:** Leave-One-Out Encoding for `Sex`
- **Validation:** 5-Fold Cross-Validation with stratified random seed
- **Blending:** Weighted average based on inverse RMSLE across folds

---

## 📉 Residual Analysis

- Out-of-fold residuals were analyzed for structure.
- A residual correction model (MLP) was trained on engineered features.
- Holdout testing showed no improvement from residual correction.
- Final submission used the raw blended predictions.

---

## 📁 File Structure

| File                     | Description                                      |
|--------------------------|--------------------------------------------------|
| `notebook.ipynb`         | Full modeling pipeline (Phases 1–5)              |
| `submission_blended.csv` | Final predictions (ready for Kaggle submission)  |
| `dataset/`               | Folder containing `train.csv`, `test.csv`, etc.  |

---

## 📤 Submission Details

- **Target Used:** `log1p(Calories)`
- **Evaluation Metric:** RMSLE
- **Cross-Validation RMSLE:** ~0.0601 (average across 5 folds)
- **Final Submission:** Weighted blend of LGBM, XGB, and CB  
- **Residual Correction:** Tested but not used (no holdout improvement)

---

## ✅ Highlights

- RMSLE-aligned modeling pipeline  
- Biologically informed feature design  
- GPU-accelerated model training  
- Clean, interpretable notebook layout  
- Final predictions verified with holdout validation

---

## 📜 License

This project is open for educational and portfolio use. Attribution is appreciated if reused.

