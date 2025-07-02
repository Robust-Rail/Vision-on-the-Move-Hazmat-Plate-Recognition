import os
import pandas as pd
import cv2
import random
from tqdm import tqdm

from annotation.video_annotator import save_frame
from annotation.converters.coco_converter import CocoWriter

distribution = [("train", 0.8), ("test", 0.1), ("val", 0.1)]
frames_path = "./data/data_faster_rcnn"
video_directory = "/deepstore/datasets/dmb/ComputerVision/ProRail/Ivg/Videos"
df = pd.read_csv("data/labels_dataframe.csv")

videos = df["Source"].unique()
available_videos = [
    v for v in os.listdir(video_directory) if v.endswith(".mp4") and v in videos
]

total_frames = df[df["Source"].isin(available_videos)]["Absolute Frame"].count()
coco_writer = CocoWriter()


def get_rnd_distribution():
    new_dist = distribution.copy()
    while new_dist:
        dist = random.choice(new_dist)
        required = int(total_frames * dist[1])
        if len(coco_writer.annotations[dist[0]]) < required:
            return dist
        new_dist.remove(dist)
    return distribution[0]


with tqdm(total=total_frames) as pbar:
    for video in available_videos:
        cap = cv2.VideoCapture(os.path.join(video_directory, video))
        video_name = os.path.splitext(video)[0]
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_num = 0

        while frame_num < total:
            ret, frame = cap.read()
            if not ret:
                break
            annotations = df[
                (df["Source"] == video) & (df["Relative Frame"] == frame_num)
            ]
            if not annotations.empty:
                dist = get_rnd_distribution()[0]
                save_frame(frame, video_name, frame_num, frames_path, dist)
                image_id = len(coco_writer.images[dist]) + 1
                coco_writer.add_image(
                    dist,
                    image_id,
                    f"{video_name}_{frame_num:05d}.jpg",
                    video_w,
                    video_h,
                )
                for _, row in annotations.iterrows():
                    coco_writer.add_annotation(dist, row, image_id)
                pbar.update(annotations.shape[0])
            frame_num += 1
        cap.release()

coco_writer.write_json(frames_path)
