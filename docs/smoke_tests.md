# Smoke Test Results

**Date:** 2026-07-02  
**OS:** Windows 10  
**Python:** 3.13 (system)

---

## Results

| # | Command | Result | Notes |
|---|---------|--------|-------|
| 1 | Notebook JSON parse (10 files in `notebooks/`) | **PASS** | All notebooks valid JSON |
| 2 | `python -c "import torch, torchvision"` | **FAIL** | `torchvision` not installed in system Python |
| 3 | `python -c "import tensorflow"` | **FAIL** | `tensorflow` not installed in system Python |
| 4 | Full notebook execute | **SKIPPED** | Not required per split policy |

---

## Summary

Notebook integrity verified. Deep-learning imports require `pip install -r requirements.txt` in a dedicated venv.

**Notebook count:** 10 `.ipynb` files (+ `README.md` in monorepo = 11 files total).
