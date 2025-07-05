import os

import cv2

from annotation.utils import get_annotation_file_name


class YOLOConverter:
    subpath = "yolo"

    def __init__(self, path):
        self.annotations_count = {"train": 0, "test": 0, "val": 0}
        self.created_annotations = []
        self.path = path

    def get_path(self):
        return os.path.join(self.path, YOLOConverter.subpath)

    def add_annotation(
        self,
        dist,
        video=None,
        frame=None,
        img_dimensions=(0, 0),
        annotation=None,
    ):
        img_width, img_height = img_dimensions
        annotation_path = os.path.join(self.get_path(), "labels", dist)
        os.makedirs(annotation_path, exist_ok=True)
        label_name = f"{get_annotation_file_name(video, frame)}.txt"
        label_path = os.path.join(annotation_path, label_name)

        lable = ""
        x_center = (annotation["XTL"] + annotation["XBR"]) / 2 / img_width
        y_center = (annotation["YTL"] + annotation["YBR"]) / 2 / img_height
        width = (annotation["XBR"] - annotation["XTL"]) / img_width
        height = (annotation["YBR"] - annotation["YTL"]) / img_height
        lable += f"0 {x_center} {y_center} {width} {height}\n"

        if (
            os.path.exists(os.path.join(annotation_path, label_path))
            and label_path in self.created_annotations
        ):
            with open(label_path, "a") as f:
                f.write(lable)
        else:
            with open(label_path, "w") as f:
                f.write(lable)

        self.annotations_count[dist] += 1
        self.created_annotations.append(label_path)

    def save_frame(self, video_name, frame_num, frame, dist):
        frames_path = os.path.join(self.get_path(), "images", dist)
        os.makedirs(frames_path, exist_ok=True)
        filename = get_annotation_file_name(video_name, frame_num)
        frame_name = f"{filename}.jpg"
        cv2.imwrite(os.path.join(frames_path, frame_name), frame)
