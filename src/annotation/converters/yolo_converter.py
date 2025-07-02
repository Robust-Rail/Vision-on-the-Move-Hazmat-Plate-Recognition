import os
import random

import cv2
import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

distribution = [("train", 0.8), ("test", 0.1), ("val", 0.1)]
path = kagglehub.dataset_download("stanislavlevendeev/hazmat-detection")
video_directory = os.environ["PATH_TO_DATA"]
print("Path to dataset files:", path)
print("Path to video files:", video_directory)
df = pd.read_csv(path + "/labels_dataframe.csv")
videos = df["Source"].unique()
videos
# get unique task source pairs from the dataframe
unique_tasks = df.drop_duplicates(subset=["Job Id", "Source", "Relative Frame"])
unique_tasks.count()
available_videos = os.listdir(video_directory)
available_videos = [video for video in available_videos if video.endswith(".mp4")]
available_videos = [video for video in available_videos if video in videos]
available_videos
total_frames = df[df["Source"].isin(available_videos)].count()["Absolute Frame"]
total_frames


def createYOLODataAnnotation(
    path=None, label_name=None, classId=None, img_width=0, img_height=0, rows=None
):
    # if already exists, skip
    if os.path.exists(os.path.join(path, label_name)):
        return
    os.makedirs(path, exist_ok=True)
    if not classId or classId == 0:
        with open(os.path.join(path, label_name), "w") as f:
            f.write(f"")
        return
    lable = ""
    for index, row in rows.iterrows():
        x_center = (row["XTL"] + row["XBR"]) / 2 / img_width
        y_center = (row["YTL"] + row["YBR"]) / 2 / img_height
        width = (row["XBR"] - row["XTL"]) / img_width
        height = (row["YBR"] - row["YTL"]) / img_height
        lable += f"0 {x_center} {y_center} {width} {height}\n"
    with open(os.path.join(path, label_name), "w") as f:
        f.write(lable)


frames_dir = path + "/images/"
os.makedirs(frames_dir, exist_ok=True)


def saveFrame(video_name="", frame_num=0, frame=None, frames_dir=frames_dir):
    # if already exists, skip
    if os.path.exists(f"{frames_dir}/{video_name}_{frame_num}.jpg"):
        return
    if frame is None:
        return
    frame_index = str(frame_num).zfill(5)
    frame_name = f"{video_name}_{frame_index}.jpg"
    cv2.imwrite(f"{frames_dir}/{frame_name}", frame)


annotations_created = {"train": 0, "test": 0, "val": 0}


def get_rnd_distribution():
    new_dist = distribution.copy()
    while len(new_dist) > 0:
        rnd_dist = random.choice(new_dist)
        required_amount = int(total_frames * rnd_dist[1])
        if required_amount >= annotations_created[rnd_dist[0]]:
            return rnd_dist
        else:
            new_dist.remove(rnd_dist)
    return distribution[0]


for name, value in distribution:
    labels_dir = os.path.join(path, "yolo", "labels", name)
    frames_dir = os.path.join(path, "yolo", "images", name)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(frames_dir, exist_ok=True)
labels_dir = os.path.join(path, "yolo", "labels")
frames_dir = os.path.join(path, "yolo", "images")
with tqdm(total=total_frames, desc="Processing") as pbar:
    for video in available_videos:
        video_path = video_directory + "/" + video
        if os.path.exists(video_path) == False:
            print(f"File {video_path} not found")
            continue
        processed_source = video.split(".")[0]
        # Open video file
        cap = cv2.VideoCapture(video_path)
        number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_labels = df[df["Source"] == video]
        frame_idx = 0
        # save each frame and .txt file with labels
        while frame_idx < number_of_frames:
            ret, frame = cap.read()  # Read each frame
            if not ret:
                break  # End of video
            img_height, img_width, _ = frame.shape
            img_height = int(img_height)
            img_width = int(img_width)
            label_frames = video_labels[video_labels["Relative Frame"] == frame_idx]
            isObject = frame_idx in video_labels["Relative Frame"].values
            if isObject:
                rnd_dist = get_rnd_distribution()
                saveFrame(
                    processed_source,
                    frame_idx,
                    frame,
                    os.path.join(frames_dir, rnd_dist[0]),
                )
                frame_index = str(frame_idx).zfill(5)
                label_name = f"{processed_source}_{frame_index}.txt"
                createYOLODataAnnotation(
                    path=os.path.join(labels_dir, rnd_dist[0]),
                    label_name=label_name,
                    classId=1 if isObject else None,
                    img_height=img_height,
                    img_width=img_width,
                    rows=label_frames,
                )
                annotations_created[rnd_dist[0]] += label_frames.shape[0]
                pbar.update(label_frames.shape[0])
            frame_idx += 1
            pbar.set_description(
                f"Processing {video}, Frame {frame_idx}/{number_of_frames}"
            )
        cap.release()
