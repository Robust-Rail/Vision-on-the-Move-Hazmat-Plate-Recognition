# src/un_detector/cli/generate_annotations.py
import os
import pandas as pd

from un_detector.annotation.converters.coco_converter import CocoConverter
from un_detector.annotation.converters.yolo_converter import YOLOConverter
from un_detector.data.preprocessing import (
    generate_prorail_annotations,
    generate_haztruck_annotations,
)


def generate_annotations(
    output_path=None,
    video_directory=None,
    df_prorail_path=None,
    df_haztruck_path=None,
    coco_writer=None,
    yolo_writer=None,
):
    if output_path is None:
        output_path = os.environ.get("PATH_TO_ANNOTATIONS")
        if output_path is None:
            raise ValueError(
                "output_path must be provided or PATH_TO_ANNOTATIONS "
                + "environment variable must be set"
            )

    if video_directory is None:
        video_directory = os.environ.get("PATH_TO_PRORAIL_VIDEODATA")
        if video_directory is None:
            raise ValueError(
                "video_directory must be provided or PATH_TO_PRORAIL_VIDEODATA "
                + "environment variable must be set"
            )

    if df_prorail_path is None:
        df_prorail_path = os.environ.get("PATH_TO_PRORAIL_CSV")
        if df_prorail_path is None:
            raise ValueError(
                "df_prorail_path must be provided or PATH_TO_PRORAIL_CSV "
                + "environment variable must be set"
            )

    if df_haztruck_path is None:
        df_haztruck_path = os.environ.get("PATH_TO_HAZTRUCK_CSV")
        if df_haztruck_path is None:
            raise ValueError(
                "df_haztruck_path must be provided or PATH_TO_HAZTRUCK_CSV "
                + "environment variable must be set"
            )

    if coco_writer is None:
        coco_writer = CocoConverter(output_path)
    if yolo_writer is None:
        yolo_writer = YOLOConverter(output_path)

    if os.path.exists(output_path):
        os.rmdir(output_path)
    os.makedirs(output_path, exist_ok=True)

    generate_prorail_annotations(
        output_path=os.path.join(output_path, "prorail"),
        video_directory=video_directory,
        df_prorail_path=df_prorail_path,
        coco_writer=coco_writer,
        yolo_writer=yolo_writer,
    )
    generate_haztruck_annotations(
        output_path=os.path.join(output_path, "haztruck"),
        df_haztruck_path=df_haztruck_path,
        coco_writer=coco_writer,
        yolo_writer=yolo_writer,
    )

    coco_writer.write_json()
