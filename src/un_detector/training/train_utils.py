# src/un_detector/training/train_utils.py
import os
import random
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple
from torch.utils.data import Subset


def collate_fn(batch):
    """
    Collate function to group images and targets for Faster R-CNN dataset loader.
    """
    return tuple(zip(*batch))


def create_subset(dataset, percentage: float) -> Subset:
    """
    Create a subset of the dataset based on the given percentage.

    Parameters:
    - dataset: The full dataset.
    - percentage: The fraction of the dataset to use (value between 0.0 and 1.0).

    Returns:
    - subset: A subset of the dataset containing the specified percentage of data.
    """
    if not (0.0 < percentage <= 1.0):
        raise ValueError("Percentage must be between 0.0 and 1.0.")

    # Determine the subset size
    total_samples = len(dataset)
    subset_size = int(total_samples * percentage)

    # Shuffle and select a random subset of indices
    indices = list(range(total_samples))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]

    return Subset(dataset, subset_indices)


def create_directory(base_path: str = "data/models") -> str:
    """
    Create a directory inside the base path named 'faster-rcnn-finetuned-{date}'
    to store models and logs. The name includes the current date and time.

    Parameters:
    - base_path: Base directory where the new directory will be created.

    Returns:
    - directory_path: Full path to the created directory.
    """
    # Get the current date and time formatted safely for directories
    current_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

    # Define the full directory path
    directory_name = f"faster-rcnn-finetuned-{current_time}"
    directory_path = os.path.join(base_path, directory_name)

    # Create the directory
    os.makedirs(directory_path, exist_ok=True)

    print(f"Directory created: {directory_path}")
    return directory_path


def save_epoch_data(directory: str, data: Dict[str, Any]) -> None:
    """
    Save training statistics for each epoch in a text file.

    Parameters:
    - directory: Path to the directory.
    - data: Dict containing epoch statistics and metrics.
    """
    log_file_path = os.path.join(directory, "training_log.txt")

    with open(log_file_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"📊 Epoch {data['epoch']} | ⏳ Time: {data['time_elapsed'][0]}m {data['time_elapsed'][1]}s | 🔄 LR: {data['learning_rate']:.6f}\n"
        )
        log_file.write(
            f"📉 Train Loss: {data['train_loss']:.4f} | 🎯 Classifier: {data['classifier_loss']:.4f} | 📦 Box Reg: {data['box_reg_loss']:.4f}\n"
        )
        log_file.write(
            f"🔍 Objectness: {data['objectness_loss']:.4f} | 🗂️ RPN Box Reg: {data['rpn_box_reg_loss']:.4f}\n"
        )
        if "val_metrics" in data and len(data["val_metrics"]) >= 6:
            val_metrics = data["val_metrics"]
            log_file.write(
                f"🧪 Validation Metrics | 🟢 mAP@IoU=0.50:0.95: {val_metrics[0]:.4f} | 🔵 mAP@IoU=0.50: {val_metrics[1]:.4f} | 🟣 mAP@IoU=0.75: {val_metrics[2]:.4f}\n"
            )
            log_file.write(
                f"📏 Small mAP: {val_metrics[3]:.4f} | 📐 Medium mAP: {val_metrics[4]:.4f} | 📏 Large mAP: {val_metrics[5]:.4f}\n"
            )
        log_file.write("\n")
