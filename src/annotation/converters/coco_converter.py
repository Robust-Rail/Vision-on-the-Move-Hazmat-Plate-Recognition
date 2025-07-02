import json
import os


class CocoConverter:
    def __init__(self):
        self.categories = [{"id": 1, "name": "hazmat"}]
        self.images = {"train": [], "test": [], "val": []}
        self.annotations = {"train": [], "test": [], "val": []}
        self.annotations_count = {"train": 0, "test": 0, "val": 0}

    def add_image(self, dist, image_id, filename, width, height):
        self.images[dist].append(
            {
                "id": image_id,
                "file_name": filename,
                "width": width,
                "height": height,
            }
        )

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

    def write_json(self, base_path):
        for dist in ["train", "val", "test"]:
            path = os.path.join(base_path, dist, "annotations")
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
        return {dist: len(self.images[dist]) for dist in self.images.keys()}
