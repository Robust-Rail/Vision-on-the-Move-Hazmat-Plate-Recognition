import math
import os
import random
from typing import Union

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import ColorJitter, GaussianBlur


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        # Convert PIL image to tensor
        image = F.to_tensor(image)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if torch.rand(1) < self.prob:
            height, width = image.shape[-2:]
            image = F.hflip(image)
            # Flip bounding boxes
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # Flip x-coordinates
            target["boxes"] = bbox
        return image, target


class RandomBrightnessCont(object):
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5):
        self.color_jitter = ColorJitter(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = self.color_jitter(image)
        return image, target


class RandomBlur(object):
    def __init__(self, kernel_size=3, p=0.5):
        self.blur = GaussianBlur(kernel_size)
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = self.blur(image)
        return image, target


class RandomRotate(object):
    def __init__(self, angle_range=10, p=0.5):
        self.angle_range = angle_range
        self.p = p

    def __call__(self, image, target):
        if random.random() >= self.p:
            return image, target

        angle = random.uniform(-self.angle_range, self.angle_range)
        original_width, original_height = self._get_image_size(image)

        # Rotate image with expansion to get new dimensions
        image_pil = image if isinstance(image, Image.Image) else F.to_pil_image(image)
        image_pil_rotated = F.rotate(image_pil, angle, expand=True)
        new_width, new_height = image_pil_rotated.size

        # Convert back to tensor if needed
        image = (
            F.to_tensor(image_pil_rotated) if isinstance(image, torch.Tensor) else image_pil_rotated
        )

        # Rotate bounding boxes
        boxes = target["boxes"]
        if len(boxes) == 0:
            return image, target

        # Compute rotation matrix with expansion offset
        cx_orig = original_width / 2
        cy_orig = original_height / 2

        # Calculate expansion offset (min_x, min_y)
        corners_original = torch.tensor(
            [
                [0, 0],
                [original_width, 0],
                [original_width, original_height],
                [0, original_height],
            ]
        )
        corners_rotated = self._rotate_points(corners_original, -angle, (cx_orig, cy_orig))
        min_x = corners_rotated[:, 0].min()
        min_y = corners_rotated[:, 1].min()

        # Rotate and translate box corners
        boxes_rotated = []
        for box in boxes:
            x1, y1, x2, y2 = box
            corners = torch.tensor([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
            corners_rot = self._rotate_points(corners, -angle, (cx_orig, cy_orig))
            # Adjust for expansion
            corners_rot -= torch.tensor([[min_x, min_y]])

            # Clamp to new image bounds
            x_min = max(0.0, corners_rot[:, 0].min().item())
            y_min = max(0.0, corners_rot[:, 1].min().item())
            x_max = min(new_width, corners_rot[:, 0].max().item())
            y_max = min(new_height, corners_rot[:, 1].max().item())

            if x_max > x_min and y_max > y_min:
                boxes_rotated.append([x_min, y_min, x_max, y_max])

        target["boxes"] = (
            torch.tensor(boxes_rotated, dtype=torch.float32)
            if boxes_rotated
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        return image, target

    def _rotate_points(self, points, angle, center):
        angle_rad = math.radians(angle)
        cos_theta = math.cos(angle_rad)
        sin_theta = math.sin(angle_rad)
        cx, cy = center

        # Translate points to origin
        translated = points - torch.tensor([[cx, cy]])

        # Apply rotation
        x_rot = translated[:, 0] * cos_theta - translated[:, 1] * sin_theta
        y_rot = translated[:, 0] * sin_theta + translated[:, 1] * cos_theta

        # Translate back
        rotated_points = torch.stack([x_rot + cx, y_rot + cy], dim=1)
        return rotated_points

    def _get_image_size(self, image):
        if isinstance(image, torch.Tensor):
            return image.shape[-1], image.shape[-2]
        elif isinstance(image, Image.Image):
            return image.size
        else:
            raise TypeError("Unsupported image type.")


class RandomZoom(object):
    def __init__(self, zoom_range=(1.0, 2.0), p=0.5):
        self.zoom_range = zoom_range
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            boxes = target["boxes"]
            if len(boxes) == 0:
                return image, target

            box_idx = random.randint(0, len(boxes) - 1)
            x1, y1, x2, y2 = boxes[box_idx].numpy()

            width, height = image.size
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            box_width, box_height = x2 - x1, y2 - y1

            zoom_factor = random.uniform(*self.zoom_range)
            crop_width = box_width / zoom_factor
            crop_height = box_height / zoom_factor

            crop_x1 = max(0, center_x - crop_width / 2)
            crop_y1 = max(0, center_y - crop_height / 2)
            crop_x2 = min(width, center_x + crop_width / 2)
            crop_y2 = min(height, center_y + crop_height / 2)

            image = image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
            target["boxes"][:, [0, 2]] -= crop_x1
            target["boxes"][:, [1, 3]] -= crop_y1
            target["boxes"][:, [0, 2]] = target["boxes"][:, [0, 2]].clamp(0, crop_x2 - crop_x1)
            target["boxes"][:, [1, 3]] = target["boxes"][:, [1, 3]].clamp(0, crop_y2 - crop_y1)

        return image, target


def get_augmented_transform(train):
    """
    Get transform pipeline with augmentations for training or validation
    """
    transforms = []

    if train:
        # Applies a series of data augmentations specifically for the training set
        transforms.extend(
            [
                RandomHorizontalFlip(0.5),  # Horizontally flips the image with a 50% probability
                # Adjusts brightness, contrast, saturation, and hue with specified ranges
                RandomBrightnessCont(
                    brightness=0.3,
                    contrast=0.4,
                    saturation=0.5,
                    hue=0.5,
                    p=0.5,  # Applies these adjustments with a 50% probability
                ),
                RandomBlur(
                    kernel_size=3, p=0.5
                ),  # Applies Gaussian blur with a kernel size of 3, 50% chance
                RandomRotate(
                    angle_range=50, p=0.5
                ),  # Rotates the image by -10 to +10 degrees, 50% chance
                RandomZoom(
                    zoom_range=(0.05, 0.99), p=1
                ),  # Zooms the image by a factor between 0.05 and 0.99, 100% chance
            ]
        )

    # Converts the image to a tensor for model input
    transforms.append(ToTensor())

    return Compose(transforms)


def visualize_augmentations(dataset, num_samples=5):
    fig, axes = plt.subplots(num_samples, 1, figsize=(8, 4 * num_samples))
    samples = min(num_samples, len(dataset))
    for idx in range(samples):
        img, target = dataset[idx]
        if isinstance(img, torch.Tensor):
            img_np = img.permute(1, 2, 0).numpy()
        else:
            img_np = np.array(img)

        axes[idx].imshow(img_np)
        axes[idx].set_title(f"Image {idx} with {len(target['boxes'])} boxes")
        axes[idx].axis("off")
        for box in target["boxes"]:
            x1, y1, x2, y2 = box
            rect = plt.Rectangle(
                (x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="r", facecolor="none"
            )
            axes[idx].add_patch(rect)
    plt.tight_layout()
    plt.show()


def generate_named_augmented_image(
    input_image_path: str,
    output_dir: str,
    output_filename_base: str,  # <<< NEW PARAMETER
    augmentation: Union[str, A.Compose],
    seed: int = None,
    quality: int = 95,
    verbose: bool = True,
    augmentation_presets: dict = {
        "rain": A.RandomRain(p=1),
        "sun_flare": A.RandomSunFlare(p=1),
        "shadow": A.RandomShadow(p=1),
        "fog": A.RandomFog(p=1),
    },
) -> str:
    """
    Generate and save a single augmented image with a specific filename.

    Parameters:
    - input_image_path: Path to source image (str)
    - output_dir: Output directory path (str) - will be created if not exists
    - output_filename_base: The base name for the output file, without the extension (str)
    - augmentation: Albumentations transform or preset name ('rain', 'fog', etc.)
    - seed: Optional random seed for reproducibility (int)
    - quality: Output JPEG quality (1-100)
    - verbose: Print progress messages

    Returns:
    The path to the saved file (str), or None if an error occurred.
    """
    # Validate inputs
    if not os.path.isfile(input_image_path):
        raise FileNotFoundError(f"Input image not found: {input_image_path}")

    if seed is not None:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load image
    image = cv2.cvtColor(cv2.imread(input_image_path), cv2.COLOR_BGR2RGB)
    # (The `base_name` from the input file is no longer needed for the output path)

    # Configure augmentation pipeline
    if isinstance(augmentation, str):
        if augmentation not in augmentation_presets:
            raise ValueError(
                f"Unknown preset: {augmentation}. "
                f"Available: {list(augmentation_presets.keys())}"
            )
        transform = A.Compose([augmentation_presets[augmentation]])
    else:
        transform = augmentation

    # Generate a single augmented image
    try:
        augmented = transform(image=image)["image"]
        if augmented is None:
            raise ValueError(
                "Augmentation returned None. Check the input image and augmentation " "parameters."
            )
        # Use the user-provided filename
        output_filename = f"{output_filename_base}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        Image.fromarray(augmented).save(
            output_path,
            quality=quality,
            optimize=True,
            subsampling=0,
        )

        if verbose:
            print(f"Saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error generating augmentation: {str(e)}")
        return None


def visualize_weather_augmentations(
    input_dir: str,
    augmentations_dir: str,
    file_extension_augmentation: str = ".jpg",
    num_images: int = 5,
):
    """
    Visualizes weather augmentations by displaying the original image
    and its augmented versions side by side.

    Args:
        input_dir (str): Directory containing the original images.
        num_images (int): Number of images to visualize. Default is 5.
        augmentations_dir (str): Directory containing the augmented images.
        file_extension_augmentation (str): File extension for the augmented images. Default is ".jpg
    """
    # Get all image files in the directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith((".jpg", ".png", ".jpeg"))]
    # Limit to num_images
    image_files = image_files[:num_images]
    possible_augmentations = ["rain", "sun_flare", "shadow", "fog"]

    for image in image_files:
        count = 1
        # Load the original image
        original_image_path = os.path.join(input_dir, image)
        original_image = Image.open(original_image_path)

        # Create a figure to display images
        fig, axes = plt.subplots(1, len(possible_augmentations) + 1, figsize=(20, 5))
        # Display the original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original Image")
        axes[0].axis("off")

        # Display each augmented image
        for i, augmentation in enumerate(possible_augmentations):
            aug_image_path = os.path.join(
                augmentations_dir, augmentation, f"{count}{file_extension_augmentation}"
            )
            if os.path.exists(aug_image_path):
                aug_image = Image.open(aug_image_path)
                axes[i + 1].imshow(aug_image)
                axes[i + 1].set_title(f"{augmentation.capitalize()} Augmentation")
                axes[i + 1].axis("off")
            else:
                print(f"Augmented image for {augmentation} not found: {aug_image_path}")

        plt.tight_layout()
        plt.show()
        count += 1
