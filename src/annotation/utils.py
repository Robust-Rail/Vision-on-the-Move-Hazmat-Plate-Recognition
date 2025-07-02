import os
import cv2
import matplotlib.pyplot as plt
import random
import json


def count_files_in_directory(directory_path):
    try:
        return len(
            [
                item
                for item in os.listdir(directory_path)
                if os.path.isfile(os.path.join(directory_path, item))
            ]
        )
    except Exception as e:
        print(f"Error: {e}")
        return 0


def check_images_in_annotations(annotation_file, image_dir, max_images=10):
    with open(annotation_file) as f:
        data = json.load(f)
        annotations = data["annotations"]
        random.shuffle(annotations)
        for annotation in annotations[:max_images]:
            image_id = annotation["image_id"]
            image_info = next(
                (img for img in data["images"] if img["id"] == image_id), None
            )
            if not image_info:
                continue
            path_im = os.path.join(image_dir, image_info["file_name"])
            image = cv2.imread(path_im)
            if image is None:
                print(f"Image not found: {path_im}")
                continue
            x, y, w, h = map(int, annotation["bbox"])
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(8, 8))
            plt.imshow(image)
            plt.gca().add_patch(
                plt.Rectangle(
                    (x, y),
                    w,
                    h,
                    edgecolor="green",
                    facecolor="none",
                    linewidth=2,
                )
            )
            plt.title(f"Image ID: {image_id}")
            plt.axis("off")
            plt.show()
