import os
import cv2
import numpy as np
import pandas as pd

from dataset_loader import TartanAirDatasetLoader
from pose_analyser import PoseAnalyzer
from config import DATASET_ROOT, SEQUENCES, ALL_LABELS_CSV

# ======================================================
# DATASET ROOT
#
# Was hardcoded to a Windows-only path in every script. Now
# comes from config.py, overridable via UAV_DATASET_ROOT.
# ======================================================

ROOT = DATASET_ROOT

pose_analyzer = PoseAnalyzer()

rows = []

orb = cv2.ORB_create(5000)
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

print("=" * 60)
print("Generating Localization Labels")
print("=" * 60)

for seq in SEQUENCES:

    print(f"\nProcessing {seq}")

    dataset = TartanAirDatasetLoader(os.path.join(ROOT, seq))

    for i in range(len(dataset) - 1):

        sample1 = dataset.get_sample(i)
        sample2 = dataset.get_sample(i + 1)

        img1 = cv2.cvtColor(sample1["rgb"], cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(sample2["rgb"], cv2.COLOR_RGB2GRAY)

        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)

        matches = 0

        if des1 is not None and des2 is not None:
            m = bf.match(des1, des2)
            matches = len(m)

        gt = pose_analyzer.translation_distance(
            sample1["pose"],
            sample2["pose"]
        )

        # ---------------------------------------
        # Estimated motion from feature matches
        # ---------------------------------------

        estimated = max(
            0.05,
            min(1.0, matches / 2000.0)
        )

        error = abs(gt - estimated)

        rows.append({
            "sequence": seq,
            "frame": i,
            "matches": matches,
            "ground_truth_distance": gt,
            "estimated_distance": estimated,
            "localization_error": error
        })

print("\nCreating dataframe...")

df = pd.DataFrame(rows)

print(df.shape)

df.to_csv(
    ALL_LABELS_CSV,
    index=False
)

print(f"\nSaved {ALL_LABELS_CSV}")