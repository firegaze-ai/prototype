import os

import cv2
import pandas as pd
import streamlit as st

from config import DATA_URL_ROOT

from yolov5.detect import run
from yolov5.utils.general import check_requirements

YOLO_LABELS_TO_STRINGS = {0: "smoke"}


def parse_yolo_label_into_dataframe(path_to_label_txt: str) -> pd.DataFrame:
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


# Run the YOLO model to detect objects.

def yolo_v5(image, confidence_threshold, overlap_threshold):
    # Load the network. Because this is cached it will only happen once.
    @st.cache(allow_output_mutation=True)
    def load_network(weights_path):
        with open(weights_path, "rb") as infile:
            net = infile.read()
        return net

    path_to_weights = os.path.join(DATA_URL_ROOT, "weights.pt")
    # weights = load_network(path_to_weights)

    import matplotlib.pyplot as plt
    # save image to disk, to be read by yolo
    path_to_tmp_image = os.path.join(DATA_URL_ROOT, 'tmpimage.jpg')
    plt.imsave(path_to_tmp_image, image)

    project_name = 'yolov5x_dataset_v100'

    # Run the YOLO neural net.
    opts = {
        'weights': path_to_weights,
        'source': path_to_tmp_image,
        'project': os.path.join(DATA_URL_ROOT, "results"),
        # 'name': project_name,
        'imgsz': [640] * 2,
        "save_txt": True,
        "nosave": True,
        "device": "CPU",
    }

    check_requirements(exclude=('tensorboard', 'thop'))
    save_dir = run(**opts)

    # TODO: implement YOLOv5 predicted label parser (from output .txt file)

    ### Dummy code follows
    print(save_dir)
    path_to_label_txt = os.path.join(save_dir, "labels",
        "tmpimage.txt")

    boxes_df = parse_yolo_label_into_dataframe(path_to_label_txt)
    boxes_df["xmin"] = (boxes_df["xmin"] * image.shape[1]).astype("int")
    boxes_df["xmax"] = (boxes_df["xmax"] * image.shape[1]).astype("int")
    boxes_df["ymin"] = (boxes_df["ymin"] * image.shape[0]).astype("int")
    boxes_df["ymax"] = (boxes_df["ymax"] * image.shape[0]).astype("int")

    return boxes_df[["xmin", "ymin", "xmax", "ymax", "labels"]]
