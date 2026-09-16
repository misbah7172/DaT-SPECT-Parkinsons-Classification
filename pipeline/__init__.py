"""DaT-SPECT Parkinson's Classification Pipeline.

Stages:
  1. preprocess  — harmonize NIfTI volumes to common spacing/matrix
  2. sbr         — extract striatal binding ratio features
  3. cnn         — train 3D CNN, extract penultimate embeddings
  4. gbm         — train LightGBM/XGBoost on SBR + CNN features
  5. calibrate   — Platt/isotonic calibration on OOF predictions
"""
