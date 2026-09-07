import cv2
import numpy as np


class FeatureExtractor:

    def brightness(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return np.mean(gray)

    def contrast(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return np.std(gray)

    def blur(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    def orb_features(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        orb = cv2.ORB_create(nfeatures=5000)

        kp, des = orb.detectAndCompute(gray, None)

        return len(kp)

    def edge_density(self, img):

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        edges = cv2.Canny(gray, 100, 200)

        density = np.sum(edges > 0) / edges.size

        return density

    def extract(self, img):

        return {
            "brightness": self.brightness(img),
            "contrast": self.contrast(img),
            "blur": self.blur(img),
            "orb_features": self.orb_features(img),
            "edge_density": self.edge_density(img)
        }