import builtins
import gc
import glob
import os
from typing import Optional

import pandas as pd
import streamlit as st
from memory_profiler import profile

from config import DATA_URL_ROOT
from tools import load_image_from_file

from yolov5_merged.detect import run, load_weights, run_with_preloaded_weights
from yolov5_merged.utils.general import check_requirements

YOLO_LABELS_TO_STRINGS = {0: "smoke"}


def parse_yolo_label_into_dataframe(path_to_label_txt: str) -> Optional[pd.DataFrame]:
    if os.path.isfile(path_to_label_txt):
        df = pd.read_csv(path_to_label_txt, header=None, sep=" ")
        df.columns = ["labels", "x_center_norm", "y_center_norm", "width_norm", "height_norm"]

        new_df = pd.DataFrame({
            "labels": df["labels"].map(YOLO_LABELS_TO_STRINGS),
            "xmin": df["x_center_norm"] - df["width_norm"] / 2,
            "xmax": df["x_center_norm"] + df["width_norm"] / 2,
            "ymin": df["y_center_norm"] - df["height_norm"] / 2,
            "ymax": df["y_center_norm"] + df["height_norm"] / 2,
        })

        # df["label"] = df["label"].map(YOLO_LABELS_TO_STRINGS)
        # TODO: map integer labels onto strings

        return new_df
    return None


# Run the YOLO model to detect objects.
# @st.cache
def yolo_v5(path_to_image, image, confidence_threshold, overlap_threshold):
    # @st.cache(hash_funcs={builtins.function: my_hash_func})
    def load_model(weights, imgsz, device):
        model, imgsz, stride, ascii, pt, classify, names, half, device = load_weights(weights, imgsz, device)
        return model, imgsz, stride, ascii, pt, classify, names, half, device

    path_to_weights = os.path.join(DATA_URL_ROOT, "weights.pt")
    imgsz = [640] * 2

    model, imgsz, stride, ascii, pt, classify, names, half, device = load_model(
        weights=path_to_weights,
        imgsz=imgsz,
        device="CPU")

    save_dir = run_with_preloaded_weights(

        model=model,  # model.pt preloaded into memory
        source=path_to_image,
        stride=stride,
        ascii=ascii,
        pt=pt,
        names=names,  # class names
        half=half,
        project=os.path.join(DATA_URL_ROOT, "results"),  # 'name': project_name,
        imgsz=imgsz,
        save_txt=True,
        nosave=True,
        device=device,
        conf_thres=confidence_threshold,  # confidence threshold
        iou_thres=overlap_threshold,  # NMS IOU threshold
    )

    check_requirements(exclude=('tensorboard', 'thop'))

    path_to_label_txt = os.path.join(save_dir, "labels",
        os.path.splitext(os.path.basename(path_to_image))[0] + ".txt")

    boxes_df = parse_yolo_label_into_dataframe(path_to_label_txt)
    if boxes_df is not None:
        boxes_df = transform_ratio_to_pixels(boxes_df, image)
        result = boxes_df[["xmin", "ymin", "xmax", "ymax", "labels"]]
        del boxes_df, image
        gc.collect()
        return result
    return None


def transform_ratio_to_pixels(boxes_df: pd.DataFrame, image) -> pd.DataFrame:
    boxes_df["xmin"] = (boxes_df["xmin"] * image.shape[1]).astype("int")
    boxes_df["xmax"] = (boxes_df["xmax"] * image.shape[1]).astype("int")
    boxes_df["ymin"] = (boxes_df["ymin"] * image.shape[0]).astype("int")
    boxes_df["ymax"] = (boxes_df["ymax"] * image.shape[0]).astype("int")
    return boxes_df


@st.cache(show_spinner=False)
def batch_parse_yolo_labels_to_csv(path_to_images_and_labels_dir: str, path_to_labels_csv: str):
    list_of_df = [(os.path.splitext(os.path.basename(txt))[0], parse_yolo_label_into_dataframe(txt)) for txt
                  in
                  glob.glob(os.path.join(path_to_images_and_labels_dir, "*.txt"))]

    # add new column "image.ext" and load image into memory
    images = []
    for filename, df in list_of_df:
        paths_to_images = glob.glob(os.path.join(path_to_images_and_labels_dir, filename + ".jp*"))
        if len(paths_to_images) > 0:
            path_to_image = paths_to_images[0]
        else:
            image_filename = ""
        df["frame"] = os.path.basename(path_to_image)
        image = load_image_from_file(path_to_image)
        df = transform_ratio_to_pixels(df, image)

    concatenated_df = pd.concat(list(zip(*list_of_df))[1]).rename(columns={'labels': 'label'})

    # Use dummy entries for cloud and fire
    concatenated_df = concatenated_df.append(
        {"label": "cloud", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "frame": "none.jpg"},
        ignore_index=True)
    concatenated_df = concatenated_df.append(
        {"label": "fire", "xmin": 0, "xmax": 1, "ymin": 0, "ymax": 1, "frame": "none.jpg"},
        ignore_index=True)

    # reorder columns to match original csv
    concatenated_df = concatenated_df[["frame", "xmin", "ymin", "xmax", "ymax", "label"]]

    concatenated_df.to_csv(path_to_labels_csv, index=False)
