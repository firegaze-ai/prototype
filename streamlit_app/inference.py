import os

import cv2
import pandas as pd
import streamlit as st

from config import DATA_URL_ROOT
from yolov5.detect import run
from yolov5.utils.general import check_requirements


def yolo_v5(image, confidence_threshold, overlap_threshold):
    # Load the network. Because this is cached it will only happen once.
    @st.cache(allow_output_mutation=True)
    def load_network(weights_path):
        with open(weights_path, "rb") as infile:
            net = infile.read()
        return net

    path_to_weights = os.path.join(DATA_URL_ROOT, "weights.pt")
    weights = load_network(path_to_weights)

    import matplotlib.pyplot as plt
    path_to_tmp_image = os.path.join(DATA_URL_ROOT, 'tmpimage.jpg')
    plt.imsave(path_to_tmp_image, image)


    # Run the YOLO neural net.
    opts = {
        'weights': path_to_weights,
        'source': path_to_tmp_image,
        'project': os.path.join(DATA_URL_ROOT, "results"),
        'name': 'yolov5x_dataset_v100',
        'imgsz': [640] * 2,
        "save_txt": True,
        "device":"CPU"
    }

    check_requirements(exclude=('tensorboard', 'thop'))
    run(**opts)

    # TODO: implement YOLOv5 predicted label parser (from output .txt file)

    ### Dummy code follows
    xmin = [100, ]
    ymin = [100, ]
    xmax = [200, ]
    ymax = [200, ]
    labels = ["smoke", ]
    ###

    boxes = pd.DataFrame({"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax, "labels": labels})
    return boxes[["xmin", "ymin", "xmax", "ymax", "labels"]]
