import os
import cv2


def save_frame(frame, video, frame_number, base_path, subpath, overwrite=True):
    new_path = os.path.join(base_path, subpath, "images")
    os.makedirs(new_path, exist_ok=True)
    formatted_frame_number = f"{frame_number:05d}"
    image_path = f"{new_path}/{video}_{formatted_frame_number}.jpg"
    if not overwrite and os.path.exists(image_path):
        return
    cv2.imwrite(image_path, frame)
