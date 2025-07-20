"""
Custom YOLO class for UN number detection that simplifies initialization and provides
common functionality for the UN number hazard plate detection project.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO


class UNNumberYOLO:
    """
    Custom YOLO class specialized for UN number hazard plate detection.

    This class wraps the Ultralytics YOLO model and provides simplified initialization,
    common training configurations, and utility methods specifically designed for
    the UN number detection project.

    Features:
    - Easy model initialization with sensible defaults
    - Automatic device detection and setup
    - Pre-configured training parameters for UN number detection
    - Built-in visualization methods
    - Simplified prediction interface
    - Model management utilities
    """

    # Common model sizes for different use cases
    MODEL_SIZES = {
        "nano": "yolo11n.pt",  # Fastest, smallest
        "small": "yolo11s.pt",  # Good balance
        "medium": "yolo11m.pt",  # Better accuracy
        "large": "yolo11l.pt",  # High accuracy
        "xlarge": "yolo11x.pt",  # Best accuracy, slowest
    }

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        model_size: str = "medium",
        device: Optional[str] = None,
        task: str = "detect",
        verbose: bool = True,
    ):
        """
        Initialize the UN Number YOLO detector.

        Args:
            model_path: Path to a trained model file (.pt). If None, uses pre-trained model.
            model_size: Size of the model ('nano', 'small', 'medium', 'large', 'xlarge').
            device: Device to run the model on ('cpu', 'cuda', 'cuda:0', etc.).
                    Auto-detected if None.
            task: Task type ('detect', 'segment', 'classify').
            verbose: Whether to print initialization details.

        Examples:
            # Use pre-trained model
            >>> detector = UNNumberYOLO()

            # Use custom trained model
            >>> detector = UNNumberYOLO('./data/yolo/best.pt')

            # Use specific model size
            >>> detector = UNNumberYOLO(model_size='xlarge')

            # Force CPU usage
            >>> detector = UNNumberYOLO(device='cpu')
        """
        self.verbose = verbose
        self.task = task

        # Setup device
        self.device = self._setup_device(device)

        # Initialize model
        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            self.model_path = str(model_path)
        else:
            if model_size not in self.MODEL_SIZES:
                valid_sizes = list(self.MODEL_SIZES.keys())
                raise ValueError(f"Invalid model size. Choose from: {valid_sizes}")
            self.model_path = self.MODEL_SIZES[model_size]

        # Load the model
        self.model = YOLO(self.model_path, task=self.task)
        self.model.to(self.device)

        if self.verbose:
            print("✅ UN Number YOLO initialized successfully")
            print(f"   📁 Model: {self.model_path}")
            print(f"   🖥️  Device: {self.device}")
            print(f"   🎯 Task: {self.task}")

    def _setup_device(self, device: Optional[str] = None) -> str:
        """Setup and return the appropriate device."""
        if device is not None:
            return device

        if torch.cuda.is_available():
            device = "cuda"
            if self.verbose:
                print(f"🚀 CUDA available. Using GPU: {torch.cuda.get_device_name()}")
        else:
            device = "cpu"
            if self.verbose:
                print("💻 CUDA not available. Using CPU.")

        return device

    @classmethod
    def load_config(cls, config_name: str = "default") -> Dict[str, Any]:
        """
        Load training configuration from YAML file.

        Args:
            config_name: Name of the configuration file (without .yaml extension).
                        Available configs: 'default', 'quick', 'high_accuracy',
                        'speed_optimized', 'robust', 'finetune'

        Returns:
            Dictionary containing configuration parameters.

        Examples:
            # Load default configuration
            >>> config = UNNumberYOLO.load_config('default')

            # Load quick training configuration
            >>> config = UNNumberYOLO.load_config('quick')
        """
        # Get the directory where this file is located
        current_dir = Path(__file__).parent.parent.parent.parent
        config_dir = current_dir / "configs"
        config_path = config_dir / f"{config_name}.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration '{config_name}' not found. ")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        return config

    @classmethod
    def list_configs(cls) -> List[str]:
        """
        List all available configuration files.

        Returns:
            List of available configuration names.
        """
        current_dir = Path(__file__).parent.parent.parent.parent
        config_dir = current_dir / "configs"

        if not config_dir.exists():
            return []

        return [f.stem for f in config_dir.glob("*.yaml")]

    def train(self, data_path: str, config_path: Optional[str] = None, **kwargs) -> Any:
        """
        Train the YOLO model with optimized parameters for UN number detection.

        Args:
            data: Path to dataset YAML file or dataset directory.
            config_path: Path to a configuration file (optional).
            **kwargs: Additional training parameters to override defaults or config.

        Returns:
            Training results object.

        Examples:
            # Basic training with default parameters
            >>> results = detector.train('data/yolo/dataset.yaml')

            # Training with predefined configuration
            >>> results = detector.train('data/yolo/dataset.yaml', config='quick')

            # Training with config and parameter overrides
            >>> results = detector.train('data/yolo/dataset.yaml',
            ...                         config='high_accuracy', epochs=150)

            # Training with custom parameters only
            >>> results = detector.train('data/yolo/dataset.yaml', epochs=20, batch=32)
        """
        # Start with default parameters
        train_params = {}

        # Load configuration if specified
        if config_path is not None:
            try:
                config_params = self.load_config(config_path)
                train_params = config_params
                if self.verbose:
                    print(f"📋 Loaded configuration: {config_path}")
            except FileNotFoundError as e:
                if self.verbose:
                    print(f"⚠️ Configuration error: {e}")
                    print("   Using default parameters instead...")

        # Apply user overrides
        train_params.update(kwargs)

        # Set device if not specified
        if train_params.get("device") is None:
            train_params["device"] = self.device

        # Ensure data path exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Dataset file not found: {data_path}")

        if self.verbose:
            print(f"🏋️ Starting training with {train_params['epochs']} epochs...")
            print(f"   📊 Dataset: {data_path}")

        # Start training
        results = self.model.train(data=str(data_path), **train_params)

        if self.verbose:
            print("✅ Training completed!")

        return results

    def predict(
        self,
        source: Union[str, Path, np.ndarray, Image.Image],
        conf: float = 0.25,
        iou: float = 0.7,
        show_boxes: bool = False,
        save: bool = False,
        **kwargs,
    ) -> Any:
        """
        Make predictions on images or videos.

        Args:
            source: Input source (image path, video path, numpy array, PIL Image).
            conf: Confidence threshold for detections.
            iou: IoU threshold for NMS.
            show_boxes: Whether to display results with bounding boxes.
            save: Whether to save results.
            **kwargs: Additional prediction parameters.

        Returns:
            Prediction results.

        Examples:
            # Predict on single image
            >>> results = detector.predict('path/to/image.jpg')

            # Predict with custom confidence
            >>> results = detector.predict('path/to/image.jpg', conf=0.5)

            # Predict and show results
            >>> results = detector.predict('path/to/image.jpg', show_boxes=True)
        """
        predict_params = {
            "conf": conf,
            "iou": iou,
            "device": self.device,
            "save": save,
            "show_boxes": show_boxes,
            "verbose": self.verbose,
        }
        predict_params.update(kwargs)

        results = self.model.predict(source, **predict_params)

        return results

    def predict_and_visualize(
        self,
        image_path: Union[str, Path],
        conf: float = 0.25,
        figsize: Tuple[int, int] = (12, 8),
        show_conf: bool = True,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Any:
        """
        Predict and visualize results with matplotlib.

        Args:
            image_path: Path to the image file.
            conf: Confidence threshold.
            figsize: Figure size for matplotlib.
            show_conf: Whether to show confidence scores on boxes.
            save_path: Path to save the visualization.

        Returns:
            Prediction results.
        """
        # Make prediction
        results = self.predict(image_path, conf=conf, save=False)

        # Load and display image
        image = cv2.imread(str(image_path))
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=figsize)
        plt.imshow(image_rgb)
        plt.axis("off")
        plt.title(f"UN Number Detection - {Path(image_path).name}")

        # Draw bounding boxes
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()

                # Draw rectangle
                rect = plt.Rectangle(
                    (x1, y1),
                    x2 - x1,
                    y2 - y1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="none",
                )
                plt.gca().add_patch(rect)

                # Add confidence text if requested
                if show_conf:
                    plt.text(
                        x1,
                        y1 - 10,
                        f"UN Code: {confidence:.2f}",
                        color="red",
                        fontsize=10,
                        fontweight="bold",
                        bbox=dict(
                            boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7
                        ),
                    )

        plt.tight_layout()

        # Save if requested
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.verbose:
                print(f"💾 Visualization saved to: {save_path}")

        plt.show()

        return results

    def evaluate(self, data: Union[str, Path], **kwargs) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.

        Args:
            data: Path to dataset YAML file.
            **kwargs: Additional evaluation parameters.

        Returns:
            Dictionary containing evaluation metrics.
        """
        eval_params = {
            "device": self.device,
            "verbose": self.verbose,
        }
        eval_params.update(kwargs)

        results = self.model.val(data=str(data), **eval_params)

        # Extract key metrics
        metrics = {
            "mAP50": results.box.map50,
            "mAP50-95": results.box.map,
            "precision": results.box.mp,
            "recall": results.box.mr,
        }

        if self.verbose:
            print("📊 Evaluation Results:")
            for metric, value in metrics.items():
                print(f"   {metric}: {value:.4f}")

        return metrics

    def save(self, path: Union[str, Path]) -> None:
        """
        Save the current model to a file.

        Args:
            path: Path where to save the model.
        """
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        self.model.save(str(save_path))

        if self.verbose:
            print(f"💾 Model saved to: {save_path}")

    def load(self, path: Union[str, Path]) -> None:
        """
        Load a model from a file.

        Args:
            path: Path to the model file.
        """
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.model = YOLO(str(model_path), task=self.task)
        self.model.to(self.device)
        self.model_path = str(model_path)

        if self.verbose:
            print(f"📂 Model loaded from: {model_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model.

        Returns:
            Dictionary containing model information.
        """
        info = {
            "model_path": self.model_path,
            "device": self.device,
            "task": self.task,
            "model_type": self.model.model.__class__.__name__,
        }

        # Try to get additional info if available
        try:
            info["num_classes"] = len(self.model.names)
            info["class_names"] = self.model.names
        except Exception:
            pass

        return info

    def print_info(self) -> None:
        """Print detailed information about the model."""
        info = self.get_model_info()

        print("🤖 UN Number YOLO Model Information:")
        print("=" * 40)
        for key, value in info.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for k, v in value.items():
                    print(f"  {k}: {v}")
            else:
                print(f"{key}: {value}")
        print("=" * 40)

    @classmethod
    def load_pretrained(
        cls,
        model_size: str = "medium",
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> "UNNumberYOLO":
        """
        Create an instance with a pre-trained model.

        Args:
            model_size: Size of the pre-trained model.
            device: Device to use.
            verbose: Whether to print information.

        Returns:
            UNNumberYOLO instance with pre-trained model.
        """
        return cls(
            model_path=None, model_size=model_size, device=device, verbose=verbose
        )

    @classmethod
    def load_custom(
        cls,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        verbose: bool = True,
    ) -> "UNNumberYOLO":
        """
        Create an instance with a custom trained model.

        Args:
            model_path: Path to the custom model file.
            device: Device to use.
            verbose: Whether to print information.

        Returns:
            UNNumberYOLO instance with custom model.
        """
        return cls(model_path=model_path, device=device, verbose=verbose)

    def __repr__(self) -> str:
        """String representation of the detector."""
        return (
            f"UNNumberYOLO(model_path='{self.model_path}', "
            f"device='{self.device}', task='{self.task}')"
        )


# Convenience functions for quick access
def create_detector(
    model_path: Optional[Union[str, Path]] = None,
    model_size: str = "medium",
    device: Optional[str] = None,
) -> UNNumberYOLO:
    """
    Quick function to create a UN Number YOLO detector.

    Args:
        model_path: Path to trained model (optional).
        model_size: Pre-trained model size if model_path is None.
        device: Device to use (auto-detected if None).

    Returns:
        UNNumberYOLO detector instance.

    Examples:
        # Create with pre-trained model
        >>> detector = create_detector()

        # Create with custom model
        >>> detector = create_detector('./data/yolo/best.pt')

        # Create with specific size
        >>> detector = create_detector(model_size='xlarge')
    """
    return UNNumberYOLO(model_path=model_path, model_size=model_size, device=device)


def load_best_model(data_dir: str = "./data/yolo") -> UNNumberYOLO:
    """
    Load the best available trained model from the data directory.

    Args:
        data_dir: Directory containing trained models.

    Returns:
        UNNumberYOLO detector with the best model.
    """
    data_path = Path(data_dir)

    # Look for common model names in order of preference
    model_candidates = [
        "best_augmented_scaled.pt",
        "best_scaled.pt",
        "best.pt",
        "last.pt",
        "yolo11x_earlystopping.pt",
        "yolo11n_trained.pt",
    ]

    for model_name in model_candidates:
        model_path = data_path / model_name
        if model_path.exists():
            print(f"🎯 Found model: {model_path}")
            return UNNumberYOLO.load_custom(model_path)

    print("⚠️ No trained models found. Using pre-trained model.")
    return UNNumberYOLO.load_pretrained()
