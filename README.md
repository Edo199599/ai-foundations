# AI-FOUNDATIONS

A hands-on repository to build solid ML/AI fundamentals through small, reproducible experiments.
The goal is to develop “engineering reflexes”: clean setup, deterministic runs, measurable results, and clear notes.

## Goals
- Build a correct mental model of training vs. generalization (overfitting, bias/variance, leakage).
- Learn to evaluate models properly (metrics, baselines, splits, validation).
- Practice simple, reproducible experiments with a clean repo structure.
- Keep a lightweight knowledge base (notes + Q/A) to support interviews and long-term retention.

## Project structure
ai-foundations/
├─ src/ # runnable scripts
├─ notes/ # learning notes + Q/A (living docs)
│ ├─ NOTES.md
│ └─ qa.md
├─ _learning/ # internal material / scratch (optional, keep lightweight)
├─ requirements.txt
├─ .gitignore
└─ README.md

## Quickstart
### 1 Create and activate a virtual environment
(bash)
python -m venv .venv
source .venv/bin/activate
### 2 Install dependencies
pip install -r requirements.txt
### 3 Run the baseline experiment
python src/train_baseline.py