import collections
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from IPython.display import Image as Ipy_Image
from IPython.display import display
from PIL import Image

from ..draw.utils import draw_box
from .image_annotator import read_image

subdirs = ["train", "test", "val"]


def display_sample_annotations_yolo(
    annotations_dir: str,
    amount: int = 5,
):
    # check if path exists and whether its a directory with folders images and labels
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Annotations directory {annotations_dir} does not exist.")
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
                [os.path.join(sub_images_dir, sub_dir) for sub_dir in os.listdir(sub_images_dir)]
            )

    for image_path in random.sample(images, min(amount, len(images))):
        # read image
        label_path = (
            image_path.replace(".jpg", ".txt").replace(".png", ".txt").replace("images", "labels")
        )
        print(f"Displaying annotations for {image_path} from {label_path}")
        image, image_width, image_height = read_image(image_path)
        if image is None:
            print(f"Image {image_path} could not be read.")
            continue
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        label_path = (
            image_path.replace(".jpg", ".txt").replace(".png", ".txt").replace("images", "labels")
        )
        if not os.path.exists(label_path):
            print(f"Label file {label_path} does not exist.")
            continue
        with open(label_path, "r") as f:
            labels = f.readlines()
            for label in labels:
                class_id, x_center, y_center, width, height = map(float, label.strip().split())
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
    subset: str = None,
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
            if img_id not in img_to_anns_sub:
                img_to_anns_sub[img_id] = []
            img_to_anns_sub[img_id].append(ann)

        img_to_anns[subdir] = img_to_anns_sub

    if subset is None:
        subset = random.choice(subdirs)
    chosen_images = list(images[subset].keys())

    for img_id in random.sample(chosen_images, min(amount, len(chosen_images))):
        img_info = images[subset][img_id]
        img_path = os.path.join(img_info["subdir"], img_info["file_name"])
        image, image_width, image_height = read_image(img_path)
        if image is None:
            print(f"Image {img_path} could not be read.")
            continue
        plt.figure(figsize=(10, 10))
        plt.imshow(image)

        anns = img_to_anns[subset].get(img_id, [])
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
        print(f"Displaying annotations for {img_path} from COCO annotations")
        plt.axis("off")
        plt.tight_layout()
        plt.show()


def display_yolo_sample(yolo_base_dir: Path, split: str = "train", image_index: int = 0):
    """
    Displays a sample image and its corresponding label from a YOLO dataset.

    Args:
        yolo_base_dir (Path): The root directory of the YOLO dataset.
        split (str): The dataset split to use (e.g., 'train', 'val', 'test').
        image_index (int): The index of the image to display from the list.
    """
    images_dir = yolo_base_dir / "images" / split
    labels_dir = yolo_base_dir / "labels" / split

    # Get a sorted list of image paths for consistent ordering
    image_paths = sorted(list(images_dir.glob("*.*")))

    if not image_paths or image_index >= len(image_paths):
        print(f"Error: No image found in '{images_dir}' at index {image_index}.")
        return

    # Select the target image and derive its label path
    sample_image_path = image_paths[image_index]
    sample_label_path = labels_dir / sample_image_path.with_suffix(".txt").name

    # --- Display Image ---
    print(f"Displaying image: {sample_image_path.name}")
    display(Ipy_Image(filename=sample_image_path, width=800))

    # --- Find and Display Label ---
    if sample_label_path.is_file():
        label_contents = sample_label_path.read_text()
        print(f"\nContents of label file '{sample_label_path.name}':\n---")
        print(label_contents.strip())
        print("---")
    else:
        print(f"\nLabel file not found at: {sample_label_path}")


def analyze_yolo_label_distribution(yolo_base_dir: Path):
    """
    Counts the number of bounding box labels in each split of a YOLO dataset
    and calculates the percentage of the total for each.

    Args:
        yolo_base_dir (Path): The root directory of the YOLO dataset.
    """
    labels_root = yolo_base_dir / "labels"
    if not labels_root.is_dir():
        print(f"Error: Labels directory not found at '{labels_root}'")
        return

    # Automatically find splits by looking for subdirectories in the labels folder
    splits = [d.name for d in labels_root.iterdir() if d.is_dir()]
    if not splits:
        print(f"No split subdirectories (e.g., 'train', 'val') found in '{labels_root}'")
        return

    print(f"Found splits: {', '.join(splits)}")
    label_counts = {}

    # Iterate through each split to count the labels
    for split in splits:
        split_dir = labels_root / split
        current_split_count = 0
        # Iterate through each .txt file in the split directory
        for label_file in split_dir.glob("*.txt"):
            with label_file.open("r") as f:
                # Count non-empty lines, as each line represents one label/box
                num_boxes = sum(1 for line in f if line.strip())
                current_split_count += num_boxes
        label_counts[split] = current_split_count

    # Calculate total and generate the report
    total_labels = sum(label_counts.values())

    print("\n--- Bounding Box Distribution Report ---")
    if total_labels == 0:
        print("No labels found across any splits.")
        return

    # Print a formatted table
    print(f"{'Split':<10} | {'Label Count':>12} | {'Percentage':>12}")
    print("-" * 40)
    for split, count in sorted(label_counts.items()):
        percentage = (count / total_labels) * 100 if total_labels > 0 else 0
        print(f"{split.capitalize():<10} | {count:>12,} | {percentage:11.2f}%")

    print("-" * 40)
    print(f"{'Total':<10} | {total_labels:>12,} | {100.00:>11.2f}%")
    print("----------------------------------------")


def find_missing_labels(yolo_base_dir: Path):
    """
    Validates a YOLO dataset by checking for images that do not have a
    corresponding label file.

    Args:
        yolo_base_dir (Path): The root directory of the YOLO dataset,
                              containing 'images' and 'labels' subdirectories.
    Returns:
        A dictionary where keys are split names and values are lists of
        image filenames that are missing labels. Returns an empty dict
        if all labels are present.
    """
    images_root = yolo_base_dir / "images"
    labels_root = yolo_base_dir / "labels"
    # Basic validation of directory structure
    if not images_root.is_dir() or not labels_root.is_dir():
        print(f"Error: Both 'images' and 'labels' directories must exist in '{yolo_base_dir}'")
        return {}

    # Automatically find splits by looking for subdirectories in the images folder
    splits = [d.name for d in images_root.iterdir() if d.is_dir()]
    if not splits:
        print(f"No split subdirectories (e.g., 'train', 'val') found in '{images_root}'")
        return {}
    print(f"🔍 Found splits to check: {', '.join(splits)}")
    missing_labels_report = collections.defaultdict(list)
    total_images_checked = 0
    total_missing = 0

    # Iterate through each split to find images
    for split in sorted(splits):
        print(f"\n--- Checking split: '{split}' ---")
        image_split_dir = images_root / split
        label_split_dir = labels_root / split
        # Check if the corresponding label directory for the split exists
        if not label_split_dir.is_dir():
            print(
                f"⚠️  Warning: Label directory for split '{split}' not found. All images in this "
                f"split will be flagged as missing labels."
            )

        image_paths = sorted(list(image_split_dir.glob("*.*")))
        if not image_paths:
            print("No images found in this split.")
            continue
        total_images_checked += len(image_paths)

        # For each image, derive the expected label path and check for its existence
        for image_path in image_paths:
            expected_label_path = label_split_dir / image_path.with_suffix(".txt").name
            if not expected_label_path.is_file():
                missing_labels_report[split].append(image_path.name)
                total_missing += 1
        if not missing_labels_report[split]:
            print(f"✅ All {len(image_paths):,} images have corresponding labels.")
        else:
            print(f"❌ Found {len(missing_labels_report[split]):,} image(s) with missing labels.")

    # --- Final Summary ---
    print("\n" + "=" * 40)
    print("      Validation Summary")
    print("=" * 40)
    if total_missing == 0:
        print(
            f"🎉 Success! All {total_images_checked:,} images across all splits have a "
            f"corresponding label file."
        )
    else:
        print(f"🚨 Found a total of {total_missing:,} image(s) with missing labels.")
        for split, missing_files in missing_labels_report.items():
            if missing_files:
                print(f"\nMissing in '{split}':")
                # Print the first 5 examples to avoid flooding the console
                for filename in missing_files[:5]:
                    print(f"  - {filename}")
                if len(missing_files) > 5:
                    print(f"  ... and {len(missing_files) - 5} more.")
    print("=" * 40)
    return dict(missing_labels_report)


def visualize_annotations_from_links(df: pd.DataFrame, num_samples: int = 3):
    """
    Fetches images from URLs in the DataFrame and draws their bounding boxes.
    This function filters out images from "Adobe Stock" and randomly selects
    a specified number of unique links to visualize.
    Args:
        df (pd.DataFrame): The haztruck_dataset DataFrame.
        num_samples (int): The maximum number of images to visualize.
    """
    # 1. Filter out Adobe Stock images
    df_filtered = df[(df["website"] != "Adobe Stock") & (df["website"].notna())].copy()
    if df_filtered.empty:
        print("No images found from sources other than Adobe Stock.")
        return

    # 2. Get a random sample of unique links to process
    unique_links = df_filtered["link"].unique()

    links_to_show = random.sample(list(unique_links), k=min(num_samples, len(unique_links)))

    # 3. Iterate through each selected link, fetch the image, and draw boxes
    from io import BytesIO

    for link in links_to_show:
        print(f"\nProcessing image from: {link}")
        try:
            # Fetch image data from the URL
            with requests.get(link, timeout=10) as response:
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")

            image_np = np.array(image)

            # Set up the plot
            fig, ax = plt.subplots(figsize=(12, 12))
            ax.imshow(image_np)
            ax.axis("off")

            # Get all annotations for this specific image
            annotations = df_filtered[df_filtered["link"] == link]

            if annotations.empty:
                ax.set_title("No annotations found for this image", fontsize=16)
                plt.tight_layout()
                plt.show()
                plt.close(fig)
                continue

            ax.set_title(f"Annotations for {annotations['image_name'].iloc[0]}", fontsize=16)

            # Draw each bounding box
            for _, row in annotations.iterrows():
                # Extract bounding box coordinates
                box = (row["box_xtl"], row["box_ytl"], row["box_xbr"], row["box_ybr"])

                # Parse the code into the format expected by draw_box
                code_parts = str(row["code"]).split("/")
                if len(code_parts) == 2:
                    display_codes = (code_parts[0], code_parts[1])
                else:
                    display_codes = (str(row["code"]), "N/A")

                # Call for side effects; do not assign the return.
                draw_box(image=image_np, ground_truth=box, codes=display_codes, ax=ax)

            plt.tight_layout()
            plt.show()

        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching image: {e}")
        except Exception as e:
            print(f"❌ An error occurred while processing the image: {e}, for link: {link}")
