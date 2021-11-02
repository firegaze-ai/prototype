import os
import shutil
import urllib
import numpy as np
import cv2
import pandas as pd

import streamlit as st

from config import EXTERNAL_DEPENDENCIES, RESULTS_DIR, LIVE_IMAGES_DIR, STATIC_IMAGES_DIR


# This file downloader demonstrates Streamlit animation.
def download_file(file_path: str):
    # Don't download the file twice. (If possible, verify the download using the file length.)
    if os.path.exists(file_path):
        if "size" not in EXTERNAL_DEPENDENCIES[file_path]:
            return
        elif os.path.getsize(file_path) == EXTERNAL_DEPENDENCIES[file_path]["size"]:
            return

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % file_path)
        progress_bar = st.progress(0)
        with open(file_path, "wb") as output_file:
            with urllib.request.urlopen(EXTERNAL_DEPENDENCIES[file_path]["url"]) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning("Downloading %s... (%6.2f/%6.2f MB)" %
                                            (file_path, counter / MEGABYTES, length / MEGABYTES))
                    progress_bar.progress(min(counter / length, 1.0))

    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


# Download a single file and make its content available as a string.
@st.cache(show_spinner=False)
def get_file_content_as_string(path):
    url = 'https://raw.githubusercontent.com/streamlit/demo-self-driving/master/' + path
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


# This function loads an image from Streamlit public repo on S3. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image_from_url(url):
    image = image_from_url(url)
    return image


# This function streams an image. Since we need the updated image on every run,
# we don't reuse the images across runs.
def stream_image_from_url(url):
    image = image_from_url(url)
    return image


def image_from_url(url):
    with urllib.request.urlopen(url) as response:
        image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = image[:, :, [2, 1, 0]]  # BGR -> RGB
    return image


# This function loads an image from disk. We use st.cache on this
# function as well, so we can reuse the images across runs.
@st.cache(show_spinner=False)
def load_image_from_file(path_to_file: str) -> np.ndarray:
    print(path_to_file)
    with open(path_to_file, "rb") as infile:
        data = np.asarray(bytearray(infile.read()), dtype="uint8")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    del data
    image = image[:, :, [2, 1, 0]]  # BGR -> RGB
    return image


@st.cache
def load_metadata(url: str)-> pd.DataFrame:
    # To make Streamlit fast, st.cache allows us to reuse computation across runs.
    # In this common pattern, we download data from an endpoint only once.
    return pd.read_csv(url)


@st.cache
def create_summary(metadata: pd.DataFrame) -> pd.DataFrame:
    # This function uses some Pandas magic to summarize the metadata Dataframe.
    one_hot_encoded = pd.get_dummies(metadata[["frame", "label"]], columns=["label"])
    summary = one_hot_encoded.groupby(["frame"]).sum().rename(columns={
        "label_smoke": "smoke (implemented)",
        "label_fire": "fire (not implemented)",
        "label_cloud": "cloud (not implemented)",
    })
    return summary


def clean_up_subfolders(path: str):
    shutil.rmtree(path)


def init_folders():
    if os.path.exists(RESULTS_DIR):
        clean_up_subfolders(RESULTS_DIR)
    os.mkdir(RESULTS_DIR)
    if not os.path.exists(LIVE_IMAGES_DIR):
        os.mkdir(LIVE_IMAGES_DIR)
    if not os.path.exists(STATIC_IMAGES_DIR):
        os.mkdir(STATIC_IMAGES_DIR)
