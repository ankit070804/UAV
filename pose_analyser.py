import numpy as np


class PoseAnalyzer:
    """
    Analyze TartanAir ground-truth poses.

    Expected pose format:
    x y z qx qy qz qw
    """

    def __init__(self):
        pass

    def translation(self, pose):
        return pose[:3]

    def quaternion(self, pose):
        return pose[3:]

    def translation_distance(self, pose1, pose2):
        t1 = self.translation(pose1)
        t2 = self.translation(pose2)

        return float(np.linalg.norm(t2 - t1))

    def motion_vector(self, pose1, pose2):
        return self.translation(pose2) - self.translation(pose1)

    def motion_features(self, pose1, pose2):

        motion = self.motion_vector(pose1, pose2)

        return {
            "dx": float(motion[0]),
            "dy": float(motion[1]),
            "dz": float(motion[2]),
            "translation_distance": float(np.linalg.norm(motion))
        }