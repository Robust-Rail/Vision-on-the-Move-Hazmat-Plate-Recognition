# src/un_detector/models/ocr.py
import re
import os
from typing import Any, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image

# Lazy import helpers for optional packages
_easyocr_available = False
try:
    import easyocr
    _easyocr_available = True
except ImportError:
    pass

_pytesseract_available = False
try:
    import pytesseract
    _pytesseract_available = True
except ImportError:
    pass

_transformers_available = False
try:
    from transformers import BitsAndBytesConfig, Idefics2ForConditionalGeneration, Idefics2Processor
    _transformers_available = True
except ImportError:
    pass


class TesseractOCR:
    """
    Wrapper for Tesseract OCR to read UN number codes from hazard placards.
    """
    def __init__(self, tesseract_cmd: Optional[str] = None):
        if not _pytesseract_available:
            raise ImportError(
                "pytesseract is required for TesseractOCR. Install it via 'pip install pytesseract'."
            )
        
        # Determine tesseract command path
        self.tesseract_cmd = tesseract_cmd or os.environ.get(
            "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        )
        if os.path.exists(self.tesseract_cmd):
            pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        else:
            # Fallback to system path if not found directly
            pass

    def _extract_un_number(self, text: str) -> str:
        un = re.findall(r"\d{2,}", text)
        return un[0] if len(un) > 0 else "00"

    def _extract_hin_number(self, text: str) -> str:
        hin = re.findall(r"\d{4,}", text)
        return hin[0] if len(hin) > 0 else "0000"

    def get_text(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, str]:
        """
        Extract UN number (upper part) and HIN number (lower part) from placard image.

        Args:
            image: PIL Image or OpenCV numpy array of the cropped placard.

        Returns:
            Tuple of (un_number, hin_number).
        """
        # Convert PIL to CV2 grayscale if needed
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image

        h, w = image_gray.shape[:2]
        if h == 0 or w == 0:
            return "00", "0000"

        # Split image horizontally in two pieces
        image_upper = image_gray[0 : int(h / 2), 0:w]
        image_lower = image_gray[int(h / 2) : h, 0:w]

        psm = 6
        option = f"--psm {psm}"

        try:
            text_un = pytesseract.image_to_string(image_upper, config=option)
            text_hin = pytesseract.image_to_string(image_lower, config=option)
            return self._extract_un_number(text_un), self._extract_hin_number(text_hin)
        except Exception as e:
            print(f"Error during Tesseract OCR: {e}")
            return "00", "0000"


class EasyOCRReader:
    """
    Wrapper for EasyOCR to read UN number codes from hazard placards.
    """
    def __init__(self, lang_list: List[str] = ["en"], gpu: bool = True):
        if not _easyocr_available:
            raise ImportError(
                "easyocr is required for EasyOCRReader. Install it via 'pip install easyocr'."
            )
        self.reader = easyocr.Reader(lang_list, gpu=gpu)

    def get_text(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, str, List[Any]]:
        """
        Extract UN number (upper part) and HIN number (lower part) from placard image using EasyOCR.

        Args:
            image: PIL Image or OpenCV numpy array of the cropped placard.

        Returns:
            Tuple of (un_number, hin_number, raw_results).
        """
        # Convert PIL to CV2 if needed (EasyOCR prefers numpy arrays)
        if isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            return "00", "0000", []

        # Predict on whole image first for debugging/raw results
        try:
            raw_results = self.reader.readtext(image, allowlist="0123456789", detail=0)
        except Exception:
            raw_results = []

        # Split image horizontally in two pieces
        image_un = image[0 : int(h / 2), 0:w]
        image_hin = image[int(h / 2) : h, 0:w]

        try:
            result_un = self.reader.readtext(image_un, allowlist="0123456789", detail=0)
            result_hin = self.reader.readtext(image_hin, allowlist="0123456789", detail=0)

            un_number = result_un[0] if len(result_un) > 0 else "00"
            hin_number = result_hin[0] if len(result_hin) > 0 else "0000"
            return un_number, hin_number, raw_results
        except Exception as e:
            print(f"Error during EasyOCR: {e}")
            return "00", "0000", []


class Idefics2OCR:
    """
    Wrapper for Idefics2 VLM model to read UN number codes from hazard placards.
    """
    DEFAULT_PROMPT = """Analyze the image and extract two key values:

    The UN number visible on the upper part of the placard.
    The code visible on the lower part of the placard, located below the horizontal line separating the two sections.

Both codes are printed in black. If either the upper or lower part cannot be detected, replace the missing value with "0." Output the extracted values as plain text, separated by a comma if multiple codes are present. No additional context or formatting is needed.

Input Examples:

    {98 {line} 4567}
    (not found, {line}, 8901)
    {101 {line} 3345}
    (not found, {line}, {not found})
    {45 {line} 2789}
    {22 {line} 5678}

Desired Output:

    98, 4567
    0, 8901
    101, 3345
    0, 0
    45, 2789
    22, 5678

Expected Transformation:

    For each input example, extract the UN number and the code below the horizontal line.
    If either part is missing (i.e., "not found"), replace it with 0.
    Output the extracted values as plain text, separated by a comma, without any additional context or formatting."""

    def __init__(
        self,
        model_name_or_path: str = "HuggingFaceM4/idefics2-8b",
        device: Optional[str] = None,
        load_in_4bit: bool = True,
    ):
        if not _transformers_available:
            raise ImportError(
                "transformers is required for Idefics2OCR. Install it via 'pip install transformers'."
            )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = Idefics2Processor.from_pretrained(model_name_or_path)

        if load_in_4bit and self.device == "cuda":
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )
        else:
            self.model = Idefics2ForConditionalGeneration.from_pretrained(
                model_name_or_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            ).to(self.device)

        # Prepare Chat Template
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.DEFAULT_PROMPT},
                    {"type": "image"},
                ],
            }
        ]
        self.prompt_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)

    def get_text(self, image: Union[Image.Image, np.ndarray]) -> Tuple[str, str]:
        """
        Extract UN number and HIN number using Idefics2 VLM.

        Args:
            image: PIL Image or OpenCV numpy array of the cropped placard.

        Returns:
            Tuple of (un_number, hin_number).
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        inputs = self.processor(images=image, text=self.prompt_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_text = self.model.generate(**inputs, max_new_tokens=500)

        generated_text = self.processor.batch_decode(generated_text, skip_special_tokens=True)[0]

        try:
            assistant_output = generated_text.split("Assistant:")[1].strip()
            # Split the output by comma to get the individual numbers
            numbers = assistant_output.split(",")
            numbers = [number.strip().replace(".", "") for number in numbers]
        except (IndexError, AttributeError) as e:
            print(f"Error parsing assistant output: {e}")
            numbers = []

        # Handle formatting fallbacks
        numbers.append("00")
        numbers.append("0000")
        un_number, hin_number = numbers[:2]

        return un_number, hin_number
