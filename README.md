# AI-for-Healthcare_AASTRA_Parth-Khera  
## MRI Dataset Preprocessing – Task 1

## Overview
This repository presents the **Task-1 preprocessing pipeline** for an MRI-based neurological disorder detection system.  
All preprocessing was executed on a **Linux GPU server accessed securely via VPN and SSH**.

The primary objective of this task is to ensure **data integrity, standardization, and reproducibility**, rather than model performance.

---

## Dataset
- **MRI Data:** Brain MRI scans in DICOM (.dcm) format organized in nested directories  
- **Metadata:** CSV file containing patient identifiers and diagnostic labels:
  - CN (Cognitively Normal)  
  - MCI (Mild Cognitive Impairment)  
  - AD (Alzheimer’s Disease)  

The CSV file serves as the **single source of ground truth** for all diagnostic labels.

---

## Preprocessing Pipeline
1. Automatic extraction of MRI data when provided as archives  
2. Loading and validation of patient-level labels from CSV  
3. Recursive detection and reading of all DICOM slices  
4. Reconstruction of subject-level 3D brain MRI volumes  
5. Application of basic background noise removal and global intensity normalization  
6. Extraction of informative central slices (3D → 2D)  
7. Assignment and numerical encoding of labels from metadata  
8. Preparation of leakage-free train/test datasets  

---

## Compliance Statement
- No data augmentation was performed  
- No labels were modified  
- No samples were added or removed  
- No class rebalancing was applied  
- No manual or disease-specific region selection was used  

All preprocessing steps were **uniformly applied**, **deterministic**, and **fully reproducible**.

---

## Summary
This pipeline converts raw brain MRI DICOM data and clinical metadata into a clean, standardized, and AI-ready dataset using a secure Linux GPU environment, while preserving the integrity of the original data.
