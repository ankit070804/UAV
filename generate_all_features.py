import os
import pandas as pd

from dataset_loader import TartanAirDatasetLoader
from feature_extractor import FeatureExtractor
from semantic_extractor import SemanticExtractor
from depth_features import DepthFeatureExtractor
from pose_analyser import PoseAnalyzer
from config import DATASET_ROOT, SEQUENCES, ALL_FEATURES_CSV

# ===========================================================
# DATASET ROOT
#
# Was hardcoded to a Windows-only path in every script. Now
# comes from config.py, overridable via UAV_DATASET_ROOT.
# ===========================================================

ROOT = DATASET_ROOT

feature_extractor = FeatureExtractor()
semantic_extractor = SemanticExtractor()
depth_extractor = DepthFeatureExtractor()
pose_analyzer = PoseAnalyzer()

all_rows = []

print("="*60)
print("Generating Features for ALL Sequences")
print("="*60)

for seq in SEQUENCES:

    print(f"\nProcessing {seq}")

    dataset = TartanAirDatasetLoader(
        os.path.join(ROOT, seq)
    )

    print("Frames :", len(dataset))

    for i in range(len(dataset)-1):

        sample = dataset.get_sample(i)

        rgb = feature_extractor.extract(sample["rgb"])
        sem = semantic_extractor.extract(sample["segmentation"])
        depth = depth_extractor.extract(sample["depth"])

        pose1 = sample["pose"]
        pose2 = dataset.get_sample(i+1)["pose"]

        motion = pose_analyzer.motion_features(
            pose1,
            pose2
        )

        row = {}

        row.update(rgb)
        row.update(sem)
        row.update(depth)
        row.update(motion)

        row["frame"] = i
        row["sequence"] = seq

        all_rows.append(row)

print("\nCreating dataframe...")

df = pd.DataFrame(all_rows)

df.fillna(0, inplace=True)

print(df.shape)

df.to_csv(
    ALL_FEATURES_CSV,
    index=False
)

print(f"\nSaved {ALL_FEATURES_CSV}")