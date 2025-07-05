import json
import os
import random

import matplotlib.pyplot as plt

from annotation.image_annotator import read_image
from draw.utils import draw_box

subdirs = ["train", "test", "val"]


def display_sample_annotations_yolo(
    annotations_dir: str,
    amount: int = 5,
):
    # check if path exists and whether its a directory with folders images and labels
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(
            f"Annotations directory {annotations_dir} does not exist."
        )
    if not os.path.isdir(annotations_dir):
        raise NotADirectoryError(f"{annotations_dir} is not a directory.")
    if not os.path.exists(os.path.join(annotations_dir, "images")):
        raise FileNotFoundError(
            f"Images directory {os.path.join(annotations_dir, 'images')} does not exist."
        )
    if not os.path.exists(os.path.join(annotations_dir, "labels")):
        raise FileNotFoundError(
            f"Labels directory {os.path.join(annotations_dir, 'labels')} does not exist."
        )
    images_dir = os.path.join(annotations_dir, "images")

    images = []
    for subdir in subdirs:
        sub_images_dir = os.path.join(images_dir, subdir)
        if os.path.exists(sub_images_dir):
            images.extend(
                [
                    os.path.join(sub_images_dir, sub_dir)
                    for sub_dir in os.listdir(sub_images_dir)
                ]
            )

    for image_path in random.sample(images, min(amount, len(images))):
        # read image
        image, image_width, image_height = read_image(image_path)
        if image is None:
            print(f"Image {image_path} could not be read.")
            continue
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        label_path = image_path.replace(".jpg", ".txt").replace("images", "labels")
        if not os.path.exists(label_path):
            print(f"Label file {label_path} does not exist.")
            continue
        print(f"Displaying annotations for {image_path} from {label_path}")
        with open(label_path, "r") as f:
            labels = f.readlines()
            for label in labels:
                class_id, x_center, y_center, width, height = map(
                    float, label.strip().split()
                )
                # Convert from YOLO format to bounding box coordinates
                x_min = int((x_center - width / 2) * image_width)
                y_min = int((y_center - height / 2) * image_height)
                x_max = int((x_center + width / 2) * image_width)
                y_max = int((y_center + height / 2) * image_height)
                # Draw the bounding box on the image using matplotlib not using drawbox

                draw_box(
                    image=image,
                    ground_truth=(x_min, y_min, x_max, y_max),
                    codes=None,
                    predicted_box=None,
                    confidence=None,
                )

        plt.axis("off")
        plt.tight_layout()
        plt.show()


def display_sample_annotations_coco(
    annotations_dir: str,
    amount: int = 5,
):
    images = {}
    img_to_anns = {}
    for subdir in subdirs:
        sub_annotations = os.path.join(
            annotations_dir, subdir, "annotations", f"instances_{subdir}.json"
        )
        with open(sub_annotations) as f:
            data = json.load(f)
        img_per_dir = {}
        for img in data["images"]:
            img["subdir"] = os.path.join(annotations_dir, subdir, "images")
            img_per_dir[img["id"]] = img
        images[subdir] = img_per_dir

        img_to_anns_sub = {}

        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_to_anns:
                img_to_anns_sub[img_id] = []
            img_to_anns_sub[img_id].append(ann)

        img_to_anns[subdir] = img_to_anns_sub

    subdir = random.choice(subdirs)
    chosen_images = list(images[subdir].keys())

    for img_id in random.sample(chosen_images, min(amount, len(chosen_images))):
        img_info = images[subdir][img_id]
        img_path = os.path.join(img_info["subdir"], img_info["file_name"])
        image, image_width, image_height = read_image(img_path)
        if image is None:
            print(f"Image {img_path} could not be read.")
            continue
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        anns = img_to_anns[subdir].get(img_id, [])
        for ann in anns:
            bbox = ann["bbox"]
            x_min = int(bbox[0])
            y_min = int(bbox[1])
            x_max = int(bbox[0] + bbox[2])
            y_max = int(bbox[1] + bbox[3])
            draw_box(
                image=image,
                ground_truth=(x_min, y_min, x_max, y_max),
                codes=None,
                predicted_box=None,
                confidence=None,
            )

        plt.axis("off")
        plt.tight_layout()
        plt.show()
