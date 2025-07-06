import cv2


def read_image(image_path: str):
    """Read an image from a given path."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_height, img_width, _ = image.shape
    return image, img_width, img_height
