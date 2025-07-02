import json
import os

import cv2

from annotation.utils import get_annotation_file_name


class CocoConverter:
    subpath = "coco"

    def __init__(self, path):
        self.categories = [{"id": 1, "name": "hazmat"}]
        self.images = {"train": [], "test": [], "val": []}
        self.annotations = {"train": [], "test": [], "val": []}
        self.annotations_count = {"train": 0, "test": 0, "val": 0}
        self.path = path

    def get_path(self):
        return os.path.join(self.path, CocoConverter.subpath)

    def add_image(self, dist, filename, width, height):
        image_id = len(self.images[dist]) + 1
        self.images[dist].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )
        return image_id

    def save_frame(self, video, frame_number, frame, dist, overwrite=True):
        new_path = os.path.join(self.get_path(), dist, "images")
        os.makedirs(new_path, exist_ok=True)
        filename = get_annotation_file_name(video, frame_number)
        image_path = f"{new_path}/{filename}.jpg"
        if not overwrite and os.path.exists(image_path):
            return
        cv2.imwrite(image_path, frame)

    def add_annotation(self, dist, annotation, image_id):
        bbox = [
            annotation["XTL"],
            annotation["YTL"],
            annotation["XBR"] - annotation["XTL"],
            annotation["YBR"] - annotation["YTL"],
        ]
        area = bbox[2] * bbox[3]
        self.annotations[dist].append(
            {
                "id": len(self.annotations[dist]) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            }
        )
        self.annotations_count[dist] += 1

    def write_json(self):
        for dist in ["train", "val", "test"]:
            path = os.path.join(
                self.get_path(), CocoConverter.subpath, dist, "annotations"
            )
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"instances_{dist}.json"), "w") as f:
                json.dump(
                    {
                        "images": self.images[dist],
                        "annotations": self.annotations[dist],
                        "categories": self.categories,
                    },
                    f,
                )

    def get_images_count(self):
        return sum(len(self.images[dist]) for dist in self.images)

    def get_annotations_count(self):
        return sum(self.annotations_count[dist] for dist in self.annotations_count)
