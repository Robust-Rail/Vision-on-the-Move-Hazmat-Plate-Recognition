import os

import pandas as pd
from tqdm import tqdm

from annotation.converters.coco_converter import CocoConverter
from annotation.converters.yolo_converter import YOLOConverter
from annotation.utils import get_annotation_file_name, get_rnd_distribution
from annotation.video_annotator import read_video


def generate_annotations(
    output_path="./data/annotations/prorail",
    video_directory="./data/processed/prorail",
    df_prorail_path="./data/labels_dataframe.csv",
    df_hazmat_path="./data/images_with_boxes.csv",
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
    generate_hazmat_annotations(
        output_path=output_path,
        df_hazmat_path=df_hazmat_path,
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


def generate_hazmat_annotations(
    output_path="./data/annotations/hazmat",
    df_hazmat_path="./data/images_with_boxes.csv",
    coco_writer=None,
    yolo_writer=None,
):
    if coco_writer is None:
        coco_writer = CocoConverter(output_path)
    if yolo_writer is None:
        yolo_writer = YOLOConverter(output_path)

    if df_hazmat_path is not None:
        df_hazmat = pd.read_csv(df_hazmat_path)
        hazmat_images = df_hazmat["image_name"].unique()
        with tqdm(total=len(hazmat_images), desc="Hazmat images") as hazmat_pbar:
            for image_name in hazmat_images:
                image_info = df_hazmat[df_hazmat["image_name"] == image_name].iloc[0]
                image_id = coco_writer.add_image(
                    "val", image_name, image_info["width"], image_info["height"]
                )
                for _, row in df_hazmat[
                    df_hazmat["image_name"] == image_name
                ].iterrows():
                    coco_writer.add_annotation("val", row, image_id)
                    yolo_writer.add_annotation(
                        "val",
                        image_name,
                        (image_info["width"], image_info["height"]),
                        row,
                    )
                hazmat_pbar.update(1)
        print(
            "\033[93m"
            "WARNING: Annotation files have been created, but the dataset may be incomplete.\n"
            "Please ensure that all hazmat images are copied to 'coco/val/images' \n"
            "and 'yolo/images/val' to finalize the dataset."
            "\033[0m"
        )

    coco_writer.write_json()
    yolo_writer.write_dataset_yaml()
