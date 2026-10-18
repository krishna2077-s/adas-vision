# 📁 Kaggle Notebook Setup Guide — SIH ADAS Vision
# =====================================================
# What goes where and what to upload

## Step-by-Step Kaggle Setup

### 1. Create 3 Kaggle Datasets (one-time upload)

#### Dataset A — `adas-src` (your Python source files)
Upload these files from your local project:
  - training/adverse_aug.py
  - training/train_local.py
  - perception/learned_road_detection.py
  - perception/lane_detection.py
  - perception/object_detection.py
  - (any other .py you want available in the notebook)
  + this dataset's metadata: kaggle/src/dataset-metadata.json

#### Dataset B — `adas-weights` (pretrained .pth)
Upload:
  - drivable_idd_full_best.pth   (~13 MB)
  + kaggle/configs/weights-dataset-metadata.json

#### Dataset C — IDD Dataset
Search Kaggle Datasets for "IDD Segmentation" and attach it.
Or upload the IDD tar archives you have locally.

---

### 2. Create the Notebook on Kaggle
1. Go to kaggle.com → Code → New Notebook
2. Upload: kaggle/notebooks/sih_adas_train.ipynb
3. Settings (right sidebar):
   - Accelerator: GPU T4 x2
   - Internet: ON (needed for pip install)
4. Add Datasets (+ Add Data button):
   - adas-src      → mounts at /kaggle/input/adas-src/
   - adas-weights  → mounts at /kaggle/input/adas-weights/
   - IDD dataset   → mounts at /kaggle/input/idd-segmentation/

### 3. Run the Notebook
   Run All → wait ~2–4 hours for full training
   Download from /kaggle/working/models/:
     - best.pth
     - drivable_idd_lraspp_768x432.onnx

### 4. Copy back to your local project
   Replace:
     drivable_idd_full_best.pth          ← best.pth
     drivable_idd_lraspp_768x432.onnx   ← the new ONNX

---

## Files to copy — quick reference

| What              | From (local)                            | Upload to Kaggle Dataset |
|-------------------|-----------------------------------------|--------------------------|
| Source code       | training/*.py + perception/*.py         | adas-src                 |
| Pretrained weights| drivable_idd_full_best.pth              | adas-weights             |
| Notebook          | kaggle/notebooks/sih_adas_train.ipynb   | New Kaggle Notebook      |

---

## Folder structure created

kaggle/
  notebooks/
    sih_adas_train.ipynb    ← THE notebook (upload to Kaggle Code)
  src/
    dataset-metadata.json   ← for adas-src Kaggle Dataset
  configs/
    weights-dataset-metadata.json  ← for adas-weights Kaggle Dataset
  outputs/                  ← (local, empty) put downloaded .pth/.onnx here
