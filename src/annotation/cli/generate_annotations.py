import os
import pandas as pd
import cv2
from tqdm import tqdm

from annotation.video_annotator import save_frame
from annotation.converters.coco_converter import CocoConverter
from annotation.utils import get_rnd_distribution

frames_path = "./data/annotations/prorail_annotations_coco"
video_directory = "./data/processed/prorail"
df = pd.read_csv("./data/labels_dataframe.csv")

videos = df["Source"].unique()
available_videos = [
    v for v in os.listdir(video_directory) if v.endswith(".mp4") and v in videos
]

total_frames = df[df["Source"].isin(available_videos)]["Absolute Frame"].count()
coco_writer = CocoConverter()


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
                dist = get_rnd_distribution(
                    total_frames, coco_writer.annotations_count
                )[0]
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
            if coco_writer.get_images_count() == total_frames:
                break
        if coco_writer.get_images_count() == total_frames:
            break
        cap.release()


coco_writer.write_json(frames_path)
