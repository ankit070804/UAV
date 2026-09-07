import cv2
import numpy as np


class DepthFeatureExtractor:
    """
    Extract statistical features from TartanAir depth images.

    NOTE:
    Currently, the depth image is treated as an encoded RGBA image.
    Once the official TartanAir depth decoding is implemented,
    these features should be computed on the decoded metric depth map.
    """

    def __init__(self):
        pass

    def preprocess(self, depth_img):
        """
        Convert RGBA depth image into a single grayscale image
        for temporary statistical analysis.
        """

        if len(depth_img.shape) == 3:

            if depth_img.shape[2] == 4:
                gray = cv2.cvtColor(depth_img, cv2.COLOR_BGRA2GRAY)

            elif depth_img.shape[2] == 3:
                gray = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)

            else:
                gray = depth_img[:, :, 0]

        else:
            gray = depth_img

        return gray.astype(np.float32)

    def mean_depth(self, depth):
        return float(np.mean(depth))

    def std_depth(self, depth):
        return float(np.std(depth))

    def min_depth(self, depth):
        return float(np.min(depth))

    def max_depth(self, depth):
        return float(np.max(depth))

    def depth_range(self, depth):
        return float(np.max(depth) - np.min(depth))

    def depth_entropy(self, depth):

        hist = cv2.calcHist(
            [depth.astype(np.uint8)],
            [0],
            None,
            [256],
            [0, 256]
        )

        hist = hist / np.sum(hist)

        hist = hist[hist > 0]

        entropy = -np.sum(hist * np.log2(hist))

        return float(entropy)

    def valid_depth_ratio(self, depth):

        valid = np.count_nonzero(depth)

        total = depth.size

        return float(valid / total)

    def extract(self, depth_img):

        depth = self.preprocess(depth_img)

        return {

            "mean_depth": self.mean_depth(depth),

            "std_depth": self.std_depth(depth),

            "min_depth": self.min_depth(depth),

            "max_depth": self.max_depth(depth),

            "depth_range": self.depth_range(depth),

            "depth_entropy": self.depth_entropy(depth),

            "valid_depth_ratio": self.valid_depth_ratio(depth)

        }