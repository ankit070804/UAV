import os
import cv2
import numpy as np


class TartanAirDatasetLoader:

    def __init__(self, dataset_path):
        r"""
        dataset_path example:
        E:\7th sem\major project 01\TartanAir\ArchVizTinyHouseDay\Data_easy\P000
        """

        self.dataset_path = dataset_path

        if not os.path.isdir(dataset_path):
            raise FileNotFoundError(
                f"Dataset folder not found: {dataset_path}\n"
                "Set the correct location via the UAV_DATASET_ROOT "
                "environment variable (see config.py)."
            )

        # Folder paths
        self.rgb_path = os.path.join(dataset_path, "image_lcam_front")
        self.depth_path = os.path.join(dataset_path, "depth_lcam_front")
        self.seg_path = os.path.join(dataset_path, "seg_lcam_front")

        # Pose file
        self.pose_path = os.path.join(dataset_path, "pose_lcam_front.txt")

        for required in (self.rgb_path, self.depth_path, self.seg_path, self.pose_path):
            if not os.path.exists(required):
                raise FileNotFoundError(
                    f"Expected TartanAir file/folder missing: {required}"
                )

        # Load pose data
        self.poses = np.loadtxt(self.pose_path)

        # Get sorted filenames
        self.rgb_files = sorted(os.listdir(self.rgb_path))
        self.depth_files = sorted(os.listdir(self.depth_path))
        self.seg_files = sorted(os.listdir(self.seg_path))

        # Check synchronization
        if not (
            len(self.rgb_files)
            == len(self.depth_files)
            == len(self.seg_files)
            == len(self.poses)
        ):
            raise ValueError(
                "RGB, Depth, Segmentation, and Pose files are not synchronized!"
            )

    def __len__(self):
        return len(self.rgb_files)

    def get_sample(self, index):

        if index < 0 or index >= len(self):
            raise IndexError("Frame index out of range.")

        # RGB
        rgb = cv2.imread(
            os.path.join(self.rgb_path, self.rgb_files[index])
        )
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Depth
        depth = cv2.imread(
            os.path.join(self.depth_path, self.depth_files[index]),
            cv2.IMREAD_UNCHANGED,
        )

        # Segmentation
        segmentation = cv2.imread(
            os.path.join(self.seg_path, self.seg_files[index]),
            cv2.IMREAD_UNCHANGED,
        )

        # Pose
        pose = self.poses[index]

        return {
            "frame_index": index,
            "rgb": rgb,
            "depth": depth,
            "segmentation": segmentation,
            "pose": pose,
        }