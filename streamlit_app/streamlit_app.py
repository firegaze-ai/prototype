# -*- coding: utf-8 -*-
# Copyright 2018-2019 Streamlit Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This demo lets you to explore the Udacity self-driving car image dataset.
# More info: https://github.com/streamlit/demo-self-driving
import glob
import shutil

import scipy
import streamlit as st
import pandas as pd
import os
import validators
import matplotlib.pyplot as plt

# Streamlit encourages well-structured code, like starting execution in a main() function.
from config import DATA_URL_ROOT, EXTERNAL_DEPENDENCIES
from inference import yolo_v5
from tools import download_file, load_image_from_url, get_file_content_as_string, load_image_from_file
from ui_elements import frame_selector_ui, object_detector_ui, draw_image_with_boxes


def main():
    # Render the readme as markdown using st.markdown.
    readme_text = st.markdown(get_file_content_as_string("instructions.md"))

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")
    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app on static dataset", "Run the app on live cams",
         "Show the source code"])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app on static dataset".')
    elif app_mode == "Show the source code":
        readme_text.empty()
        st.code(get_file_content_as_string("app.py"))
    elif app_mode == "Run the app on static dataset":
        readme_text.empty()
        run_the_app_static()
    elif app_mode == "Run the app on live cams":
        readme_text.empty()
        run_the_app_live()


def clean_up_subfolders(path: str):
    shutil.rmtree(path)


# This is the main app app itself, which appears when the user selects "Run the app on static dataset".
def run_the_app_static():
    clean_up_subfolders(os.path.join(DATA_URL_ROOT, "results"))

    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache
    def load_metadata(url):
        return pd.read_csv(url)

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
            "label_smoke": "smoke",
            "label_fire": "fire",
            "label_cloud": "cloud",
        })
        return summary

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv"))
    summary = create_summary(metadata)

    # Uncomment these lines to peek at these DataFrames.
    # st.write('## Metadata', metadata[:1000], '## Summary', summary[:1000])

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load the image from S3.
    image_url = os.path.join(DATA_URL_ROOT, selected_frame)
    if os.path.isfile(image_url):
        image = load_image_from_file(image_url)
    elif validators.url(image_url):
        image = load_image_from_url(image_url)

    # Add boxes for objects on the image. These are the boxes for the ground image.
    boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    draw_image_with_boxes(image, boxes, "Ground Truth",
        "**Human-annotated data** (frame `%i`)" % selected_frame_index)

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    yolo_boxes = yolo_v5(image_url, image, confidence_threshold, overlap_threshold)
    draw_image_with_boxes(image, yolo_boxes, "Real-time Computer Vision",
        "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)" % (
            overlap_threshold, confidence_threshold))


# This is the main app app itself, which appears when the user selects "Run the app on static dataset".
def run_the_app_live():
    clean_up_subfolders(os.path.join(DATA_URL_ROOT, "results"))


    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    @st.cache
    def load_metadata(url):
        return pd.read_csv(url)

    # This function uses some Pandas magic to summarize the metadata Dataframe.
    @st.cache
    def create_summary(metadata):
        one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
        summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
            "label_smoke": "smoke",
            "label_fire": "fire",
            "label_cloud": "cloud",
        })
        return summary

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!
    metadata = load_metadata(os.path.join(DATA_URL_ROOT, "labels.csv"))
    summary = create_summary(metadata)

    # Uncomment these lines to peek at these DataFrames.
    # st.write('## Metadata', metadata[:1000], '## Summary', summary[:1000])

    # Draw the UI elements to search for objects (pedestrians, cars, etc.)
    selected_frame_index, selected_frame = frame_selector_ui(summary)
    if selected_frame_index == None:
        st.error("No frames fit the criteria. Please select different label or number.")
        return

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()

    # Load the images

    urls = {
        "amfikleia.jpg": "https://www.kopaida.gr/cams/amfikleia.jpg",
        "arachova.jpg": "https://www.kopaida.gr/cams/arachova.jpg",
        "kopaida.jpg": "https://www.kopaida.gr/cams/adcam3.jpg",
        # "penteli.jpg": "https://penteli.meteo.gr/stations/penteli/webcam/meteocam1.jpg",
        "prespes.jpg": "https://www.stravon.gr/meteocams/prespes/",
        "aridaia.jpg": "https://www.stravon.gr/meteocams/aridaia/",
        "fylakti.jpg": "https://www.fylakti.com/image3.jpg",
        "elati.jpg": "https://cams.elaticam.com/output/elaticam/index.php",
        # "theodoriana": "https://penteli.meteo.gr/stations/theodoriana/theodoriana1.jpg",
        "gardiki.jpg": "https://www.gardiki-live.gr/livecam/live000M.jpg",
        # "metsovo": "https://penteli.meteo.gr/stations/metsovo/webcam.jpg",
        # "terracom.jpg": "https://www.terracom.gr/sites/default/files/pictures/cam_01/cam05.jpg?1541233685360",
    }

    images = []
    for image_name, image_url in urls.items():
        path_to_image = os.path.join(DATA_URL_ROOT, image_name)
        if os.path.isfile(image_url):
            image = load_image_from_file(image_url)
        elif validators.url(image_url):
            image = load_image_from_url(image_url)
            if os.path.isfile(path_to_image):
                os.remove(path_to_image)
            plt.imsave(path_to_image, image)

        images.append((path_to_image, image, image_name))

    # Add boxes for objects on the image. These are the boxes for the ground image.
    for path_to_image, image, image_name in images:
        yolo_boxes = yolo_v5(path_to_image, image, confidence_threshold, overlap_threshold)
        draw_image_with_boxes(image, yolo_boxes, "{}".format(image_name),
            "**YOLO v5 Model** (overlap `%3.1f`) (confidence `%3.1f`)" % (
                overlap_threshold, confidence_threshold))


if __name__ == "__main__":
    main()
