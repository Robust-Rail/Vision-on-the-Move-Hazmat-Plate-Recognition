import matplotlib.pyplot as plt

LINEWIDTH_PREDICTION = 2
LINEWIDTH_GROUND_TRUTH = 4
FONT_SIZE = 15

PREDICTION_COLOR = "#CC79A7"  # Color for predicted bounding box
GROUND_TRUTH_COLOR = "#0072B2"  # Color for ground truth bounding box


def draw_box(
    image,
    predicted_box: tuple[float, float, float, float] = None,
    ground_truth: tuple[float, float, float, float] = None,
    codes=(0, 0),
    confidence: float = None,
):
    """
    Draws a bounding box on the image.

    Args:
        image (numpy.ndarray):
            The image on which to draw the box.
        predicted_box (tuple):
            A tuple containing the coordinates of the predicted bounding box (x, y, width, height).
        ground_truth (tuple):
            A tuple containing the coordinates of the ground truth bounding box (x, y, width).
        codes (tuple):
            A tuple containing two integers representing color codes for the predicted and
            ground truth boxes.

    Returns:
        numpy.ndarray: The image with the drawn bounding boxes.
    """
    if ground_truth is not None:
        x_gt, y_gt, w_gt, h_gt = get_coordinates(ground_truth)
        plt.gca().add_patch(
            plt.Rectangle(
                (x_gt, y_gt),
                w_gt,
                h_gt,
                edgecolor=GROUND_TRUTH_COLOR,
                facecolor="none",
                linewidth=LINEWIDTH_GROUND_TRUTH,
            )
        )
        plt.text(
            x_gt,
            y_gt - 5,
            "Ground Truth",
            color="white",
            fontsize=FONT_SIZE,
            bbox=dict(facecolor=GROUND_TRUTH_COLOR, alpha=0.7, edgecolor="none", pad=1),
        )
    if predicted_box is not None:
        x_pred, y_pred, w_pred, h_pred = get_coordinates(predicted_box)
        plt.gca().add_patch(
            plt.Rectangle(
                (x_pred, y_pred),
                w_pred,
                h_pred,
                edgecolor=PREDICTION_COLOR,
                facecolor="none",
                linewidth=2,
            )
        )
        text_to_display = ""
        if (codes is not None) and (len(codes) == 2):
            text_to_display = f"RID/ADR: {codes[0]}\nUN: {codes[1]}"
        if confidence is not None:
            text_to_display += f"\nYOLO-conf: {confidence:.2f}"
        plt.text(
            x_pred,
            y_pred - 5,
            text_to_display,
            color="white",
            fontsize=FONT_SIZE,
            bbox=dict(facecolor=PREDICTION_COLOR, alpha=0.7, edgecolor="none", pad=1),
        )

    return image


def get_coordinates(
    box: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """
    Converts a bounding box from (x1, y1, x2, y2) format to (x, y, width, height) format.
    Args:
        box (tuple): A tuple containing the coordinates of the bounding box (x1, y1, x2, y2).
    Returns:
        tuple: A tuple containing the coordinates in (x, y, width, height) format.
    """
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    return (x1, y1, w, h)
