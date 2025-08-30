import os

import pandas as pd
import requests
from tqdm import tqdm

from annotation.converters.coco_converter import CocoConverter
from annotation.converters.yolo_converter import YOLOConverter
from annotation.utils import get_annotation_file_name, get_rnd_distribution
from annotation.video_annotator import read_video


def generate_annotations(
    output_path="./data/annotations/prorail",
    video_directory="./data/processed/prorail",
    df_prorail_path="./data/labels_dataframe.csv",
    df_haztruck_path="./data/images_with_boxes.csv",
    coco_writer=None,
    yolo_writer=None,
):
    if coco_writer is None:
        coco_writer = CocoConverter(output_path)
    if yolo_writer is None:
        yolo_writer = YOLOConverter(output_path)

    if os.path.exists(output_path):
        os.rmdir(output_path)
    os.makedirs(output_path, exist_ok=True)

    generate_prorail_annotations(
        output_path=output_path,
        video_directory=video_directory,
        df_prorail_path=df_prorail_path,
        coco_writer=coco_writer,
        yolo_writer=yolo_writer,
    )
    generate_haztruck_annotations(
        output_path=output_path,
        df_haztruck_path=df_haztruck_path,
        coco_writer=coco_writer,
        yolo_writer=yolo_writer,
    )

    coco_writer.write_json()


def generate_prorail_annotations(
    output_path="./data/annotations/prorail",
    video_directory="./data/processed/prorail",
    df_prorail_path="./data/labels_dataframe.csv",
    coco_writer=None,
    yolo_writer=None,
):
    if coco_writer is None:
        coco_writer = CocoConverter(output_path)
    if yolo_writer is None:
        yolo_writer = YOLOConverter(output_path)

    if video_directory is not None and df_prorail_path is not None:
        df_prorail = pd.read_csv(df_prorail_path)

        videos = df_prorail["source"].unique()
        available_videos = [
            v for v in os.listdir(video_directory) if v.endswith(".mp4") and v in videos
        ]

        total_frames = df_prorail[df_prorail["source"].isin(available_videos)][
            "absolute_frame"
        ].count()

        with tqdm(total=total_frames) as pbar:
            for video in available_videos:
                cap, video_name, video_w, video_h, frames_count = read_video(
                    video_directory, video
                )
                frame_num = 0

                pbar.set_description(f"Processing {video_name}")

                while frame_num < frames_count:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    annotations = df_prorail[
                        (df_prorail["source"] == video)
                        & (df_prorail["relative_frame"] == frame_num)
                    ]
                    if not annotations.empty:
                        dist = get_rnd_distribution(
                            total_frames, coco_writer.annotations_count
                        )[0]

                        coco_writer.save_frame(video_name, frame_num, frame, dist)
                        yolo_writer.save_frame(video_name, frame_num, frame, dist)

                        image_name = get_annotation_file_name(video_name, frame_num)

                        image_id = coco_writer.add_image(
                            dist,
                            f"{image_name}.jpg",
                            video_w,
                            video_h,
                        )

                        for _, row in annotations.iterrows():
                            coco_writer.add_annotation(dist, row, image_id)
                            yolo_writer.add_annotation(
                                dist,
                                image_name,
                                (video_w, video_h),
                                row,
                            )

                        pbar.update(annotations.shape[0])

                    frame_num += 1
                    if coco_writer.get_annotations_count() == total_frames:
                        break
                if coco_writer.get_annotations_count() == total_frames:
                    break
                cap.release()

    coco_writer.write_json()
    yolo_writer.write_dataset_yaml()


def generate_haztruck_annotations(
    output_path="./data/annotations/haztruck",
    df_haztruck_path="./data/images_with_boxes.csv",
    coco_writer=None,
    yolo_writer=None,
):
    if coco_writer is None:
        coco_writer = CocoConverter(output_path)
    if yolo_writer is None:
        yolo_writer = YOLOConverter(output_path)

    if df_haztruck_path is not None:
        df_haztruck = pd.read_csv(df_haztruck_path)
        haztruck_images = df_haztruck["image_name"].unique()
        with tqdm(total=len(haztruck_images), desc="Hazmat images") as haztruck_pbar:
            for image_name in haztruck_images:
                image_info = df_haztruck[df_haztruck["image_name"] == image_name].iloc[
                    0
                ]
                width, height = map(int, image_info["resolution"].split("x"))
                image_id = coco_writer.add_image("val", image_name, width, height)
                link_to_image = image_info["link"]
                download_image(
                    image_name,
                    link_to_image,
                    [
                        os.path.join(
                            output_path,
                            coco_writer.subpath,
                            "val",
                            "images",
                            image_name,
                        ),
                        os.path.join(
                            output_path,
                            yolo_writer.subpath,
                            "images",
                            "val",
                            image_name,
                        ),
                    ],
                )

                for _, row in df_haztruck[
                    df_haztruck["image_name"] == image_name
                ].iterrows():
                    row["xtl"] = row["box_xtl"]
                    row["ytl"] = row["box_ytl"]
                    row["xbr"] = row["box_xbr"]
                    row["ybr"] = row["box_ybr"]
                    coco_writer.add_annotation("val", row, image_id)
                    yolo_writer.add_annotation(
                        "val",
                        image_name,
                        (width, height),
                        row,
                    )
                haztruck_pbar.update(1)
        print(
            "\033[93m"
            "WARNING: Annotation files have been created, but the dataset may be incomplete.\n"
            "Please ensure that all haztruck images are copied to 'coco/val/images' \n"
            "and 'yolo/images/val' to finalize the dataset."
            "\033[0m"
        )

    coco_writer.write_json()
    yolo_writer.write_dataset_yaml()


def download_image(image_name, url, output_paths):
    if pd.isna(url) or not str(url).strip().startswith(("http://", "https://")):
        print(f"Invalid URL provided for image {image_name}.")
        return
    images_downloaded = 0
    for output_path in output_paths:
        if os.path.exists(output_path):
            images_downloaded += 1
        else:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if images_downloaded == len(output_paths):
        return

    response = requests.get(url)
    if response.status_code == 200:
        for output_path in output_paths:
            with open(output_path, "wb") as f:
                f.write(response.content)
    else:
        print(f"Failed to download image from {url}")

    print(f"Downloaded image from {url}")
