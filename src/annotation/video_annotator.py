import os

import cv2


def read_video(video_dir, video_name):
    cap = cv2.VideoCapture(os.path.join(video_dir, video_name))
    video_name = os.path.splitext(video_name)[0]
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, video_name, video_w, video_h, frames_count
