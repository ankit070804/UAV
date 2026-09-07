import numpy as np
import cv2


class SemanticExtractor:
    """
    Extract semantic class percentages from segmentation images.

    Current Version:
        - Counts unique pixel values/classes
        - Computes percentage occupied by each class

    Later:
        - Replace class IDs with actual names from TartanAir labels.
    """

    def __init__(self):
        pass

    def extract(self, seg_img):

        # -------------------------------------------------
        # Convert RGB segmentation to grayscale labels
        # (If segmentation is already single-channel,
        # this line has no effect.)
        # -------------------------------------------------

        if len(seg_img.shape) == 3:
            gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = seg_img

        total_pixels = gray.size

        unique, counts = np.unique(gray, return_counts=True)

        features = {}

        for cls, cnt in zip(unique, counts):
            features[f"class_{int(cls)}_percent"] = (
                cnt / total_pixels
            ) * 100

        return features

    def inspect_classes(self, seg_img):

        if len(seg_img.shape) == 3:
            gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = seg_img

        unique = np.unique(gray)

        print("=" * 50)
        print("Semantic Classes Found")
        print("=" * 50)

        print(unique)
        print("Total Classes:", len(unique))