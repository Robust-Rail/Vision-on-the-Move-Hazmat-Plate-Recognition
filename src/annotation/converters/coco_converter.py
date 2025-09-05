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
        image_id = int(len(self.images[dist]) + 1)
        filename = filename.split("/")[1] if "/" in filename else filename
        self.images[dist].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": int(width),
                "height": int(height),
            }
        )
        return image_id

    def add_annotation(self, dist, annotation, image_id):
        bbox = [
            float(annotation["xtl"]),
            float(annotation["ytl"]),
            float(annotation["xbr"]) - float(annotation["xtl"]),
            float(annotation["ybr"]) - float(annotation["ytl"]),
        ]
        area = float(bbox[2] * bbox[3])
        self.annotations[dist].append(
            {
                "id": int(len(self.annotations[dist]) + 1),
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": [float(x) for x in bbox],
                "area": area,
                "iscrowd": 0,
            }
        )
        self.annotations_count[dist] += 1

    def save_frame(self, video, frame_number, frame, dist, overwrite=True):
        new_path = os.path.join(self.get_path(), dist, "images")
        os.makedirs(new_path, exist_ok=True)
        filename = get_annotation_file_name(video, frame_number)
        image_path = f"{new_path}/{filename}.jpg"
        if not overwrite and os.path.exists(image_path):
            return
        cv2.imwrite(image_path, frame)

    def write_json(self):
        for dist in ["train", "val", "test"]:
            path = os.path.join(self.get_path(), dist, "annotations")
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
