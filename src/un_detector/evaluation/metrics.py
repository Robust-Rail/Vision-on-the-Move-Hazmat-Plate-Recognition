# src/un_detector/evaluation/metrics.py
from typing import List, Union

try:
    import Levenshtein
    _levenshtein_available = True
except ImportError:
    _levenshtein_available = False


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes the Levenshtein distance between two strings.
    Uses the Levenshtein library if available, otherwise falls back to a DP implementation.
    """
    if _levenshtein_available:
        return Levenshtein.distance(s1, s2)

    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j] + 1,      # Deletion
                    dp[i][j - 1] + 1,      # Insertion
                    dp[i - 1][j - 1] + 1,  # Substitution
                )
    return dp[m][n]


def calculate_cer(gt: str, ocr: str) -> float:
    """
    Calculate the Character Error Rate (CER) between ground truth and OCR prediction.
    """
    gt = str(gt)
    ocr = str(ocr)
    if not gt:
        return 1.0 if ocr else 0.0
    return levenshtein_distance(gt, ocr) / len(gt)


def calculate_wer(gt: str, ocr: str) -> float:
    """
    Calculate the Word Error Rate (WER) between ground truth and OCR prediction.
    """
    gt_words = str(gt).split()
    ocr_words = str(ocr).split()
    if not gt_words:
        return 1.0 if ocr_words else 0.0
    return levenshtein_distance(" ".join(gt_words), " ".join(ocr_words)) / len(gt_words)
