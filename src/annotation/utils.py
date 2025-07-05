import os
import cv2
import matplotlib.pyplot as plt
import random
import json

distribution = [("train", 0.8), ("test", 0.1), ("val", 0.1)]


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


def get_rnd_distribution(total_frames=0, annotation_dist=None):
    new_dist = distribution.copy()
    if annotation_dist is None or total_frames == 0:
        return random.choice(new_dist)
    while new_dist:
        dist = random.choice(new_dist)
        required = int(total_frames * dist[1])
        if annotation_dist[dist[0]] < required:
            return dist
        new_dist.remove(dist)
    return distribution[0]


def get_annotation_file_name(video_name, frame_idx):
    formatted_frame_number = f"{frame_idx:05d}"
    return f"{video_name}_{formatted_frame_number}"


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
