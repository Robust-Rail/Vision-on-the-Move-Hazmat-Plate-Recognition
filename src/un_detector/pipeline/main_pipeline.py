# src/un_detector/pipeline/main_pipeline.py
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from un_detector.models.yolo_detector import UNNumberYOLO
from un_detector.models.ocr import Idefics2OCR, EasyOCRReader, TesseractOCR


class UNNumberPipeline:
    """
    Orchestrated pipeline combining YOLO hazard plate detection and
    VLM/OCR-based reading of Hazard Identification Numbers (HIN) and UN numbers.
    """
    def __init__(self,
                 yolo_model_path: Optional[str] = None,
                 ocr_engine_type: str = "idefics2",
                 ocr_engine_instance: Optional[Any] = None,
                 un_labels_csv: Optional[str] = None,
                 device: Optional[str] = None):
        """
        Initialize the orchestrated pipeline.

        Args:
            yolo_model_path: Path to trained YOLO model. If None, uses default best model.
            ocr_engine_type: Type of OCR to use ('idefics2', 'easyocr', 'tesseract').
            ocr_engine_instance: Pre-initialized OCR wrapper instance to reuse.
            un_labels_csv: Path to un-number-labels.csv. Auto-located if None.
            device: Computing device ('cpu', 'cuda', etc.).
        """
        self.device = device
        
        # 1. Initialize detector
        if yolo_model_path is None:
            # Look for local custom model or fall back to pretrained
            possible_yolo_paths = [
                "./data/yolo/yolo11x_earlystopping.pt",
                "../data/yolo/yolo11x_earlystopping.pt",
                "kaggle_dataset/yolo11x_earlystopping.pt",
                "checkpoints/yolo11x_earlystopping.pt"
            ]
            for path in possible_yolo_paths:
                if os.path.exists(path):
                    yolo_model_path = path
                    break
                    
        self.detector = UNNumberYOLO(
            model_path=yolo_model_path,
            device=self.device,
            verbose=False
        )

        # 2. Initialize OCR engine
        if ocr_engine_instance is not None:
            self.ocr_engine = ocr_engine_instance
        else:
            engine_type = ocr_engine_type.lower()
            if engine_type == "idefics2":
                self.ocr_engine = Idefics2OCR(device=self.device)
            elif engine_type == "easyocr":
                self.ocr_engine = EasyOCRReader(gpu=(self.device != "cpu"))
            elif engine_type == "tesseract":
                self.ocr_engine = TesseractOCR()
            else:
                raise ValueError(f"Unsupported OCR engine type: {ocr_engine_type}")

        # 3. Load UN numbers database
        if un_labels_csv is None:
            possible_csv_paths = [
                "./data/un-number-labels.csv",
                "./kaggle_dataset/un-number-labels.csv",
                "../kaggle_dataset/un-number-labels.csv",
                "../data/un-number-labels.csv"
            ]
            for path in possible_csv_paths:
                if os.path.exists(path):
                    un_labels_csv = path
                    break
        
        self.df_un_numbers = None
        if un_labels_csv and os.path.exists(un_labels_csv):
            try:
                self.df_un_numbers = pd.read_csv(un_labels_csv)
                # Normalize column names
                self.df_un_numbers.columns = [c.lower() for c in self.df_un_numbers.columns]
                print(f"📋 Loaded UN number description database from: {un_labels_csv}")
            except Exception as e:
                print(f"⚠️ Error loading UN descriptions: {e}")
        else:
            print("⚠️ Warning: UN number labels CSV not found. Descriptions will not be available.")

    def get_description(self, un_number: str) -> Optional[str]:
        """Look up the description of a 4-digit UN number from the database."""
        if self.df_un_numbers is None:
            return None

        try:
            un_num_int = int(un_number)
        except ValueError:
            return None

        # Look up by normalized column 'number'
        match = self.df_un_numbers[self.df_un_numbers["number"] == un_num_int]
        if not match.empty:
            return match["description"].values[0]
        return None

    def run(self,
            image_path_or_array: Union[str, Path, np.ndarray, Image.Image],
            conf_threshold: float = 0.25,
            iou_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Run the complete pipeline: Detect placards -> Crop -> OCR -> Query Description.

        Args:
            image_path_or_array: Target image.
            conf_threshold: Bounding box detection confidence threshold.
            iou_threshold: Intersection over Union threshold for NMS.

        Returns:
            List of dictionaries containing detection bounding box, confidence,
            HIN number, UN number, and material description.
        """
        # Load image array for cropping
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
        elif isinstance(image_path_or_array, Image.Image):
            img = cv2.cvtColor(np.array(image_path_or_array), cv2.COLOR_RGB2BGR)
        else:
            img = image_path_or_array.copy()

        # Get YOLO detections
        results = self.detector.predict(img, conf=conf_threshold, iou=iou_threshold)
        predictions = []

        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                # Convert coords to list of floats
                bbox = box.xyxy[0].cpu().numpy().tolist()
                confidence = float(box.conf[0].cpu().numpy())
                
                # Crop region
                x1, y1, x2, y2 = map(int, bbox)
                # Boundary clamping
                h_img, w_img = img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                
                cropped_img = img[y1:y2, x1:x2]
                if cropped_img.size == 0:
                    continue

                # Run OCR
                # EasyOCR returns (un, hin, raw)
                if isinstance(self.ocr_engine, EasyOCRReader):
                    hin_number, un_number, _ = self.ocr_engine.get_text(cropped_img)
                else:
                    hin_number, un_number = self.ocr_engine.get_text(cropped_img)

                # Get description based on the 4-digit UN number (un_number)
                description = self.get_description(un_number)

                predictions.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "hin_number": hin_number,
                    "un_number": un_number,
                    "description": description
                })

        return predictions

    def visualize_predictions(self,
                              image_path_or_array: Union[str, Path, np.ndarray, Image.Image],
                              predictions: List[Dict[str, Any]],
                              save_path: Optional[str] = None) -> None:
        """
        Draw bounding boxes, confidence, OCR codes and material descriptions
        on the source image.
        """
        # Load image array for plotting
        if isinstance(image_path_or_array, (str, Path)):
            img = cv2.imread(str(image_path_or_array))
        elif isinstance(image_path_or_array, Image.Image):
            img = np.array(image_path_or_array)
        else:
            img = image_path_or_array.copy()

        # Convert to RGB if loaded using cv2
        if isinstance(image_path_or_array, (str, Path, np.ndarray)):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 9))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"UN Number Pipeline Analysis ({len(predictions)} placard(s) found)")

        ax = plt.gca()
        for pred in predictions:
            x1, y1, x2, y2 = pred["bbox"]
            conf = pred["confidence"]
            hin = pred["hin_number"]
            un = pred["un_number"]
            desc = pred["description"] or "Unknown material"

            # Draw rectangle
            rect = plt.Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                linewidth=3,
                edgecolor="lime",
                facecolor="none"
            )
            ax.add_patch(rect)

            # Draw text label overlay
            label_text = f"HIN: {hin}\nUN: {un}\n[{desc}]"
            ax.text(
                x1,
                y1 - 15,
                label_text,
                color="white",
                fontsize=9,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="darkgreen", alpha=0.85)
            )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"💾 Visualization saved to: {save_path}")
        plt.show()
