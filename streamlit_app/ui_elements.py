from typing import Union, Tuple

import altair as alt
import pandas as pd
import streamlit as st
import numpy as np

# This sidebar UI is a little search engine to find certain object types.
from config import GARBAGE_COLLECT


def frame_selector_ui(summary: pd.DataFrame) -> Tuple[int, str]:
    st.sidebar.markdown("# Frame")

    # The user can pick which type of object to search for.
    object_type = st.sidebar.selectbox("Search for which objects?", summary.columns, 2)
    selected_frames = get_selected_frames(summary, object_type)
    if len(selected_frames) < 1:
        return None, None

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider("Choose a frame (index)", 0, len(selected_frames) - 1, 0)

    if 0:
        # Draw an altair chart in the sidebar with information on the frame.
        objects_per_frame = summary.loc[selected_frames, object_type].reset_index(drop=True).reset_index()
        chart = alt.Chart(objects_per_frame, height=120).mark_area().encode(
            alt.X("index:Q", scale=alt.Scale(nice=False)),
            alt.Y("%s:Q" % object_type))
        selected_frame_df = pd.DataFrame({"selected_frame": [selected_frame_index]})
        vline = alt.Chart(selected_frame_df).mark_rule(color="red").encode(x="selected_frame")
        st.sidebar.altair_chart(alt.layer(chart, vline))

    selected_frame = selected_frames[selected_frame_index]
    return selected_frame_index, selected_frame


# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui() -> Tuple[float, float]:
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold


# Select frames based on the selection in the sidebar
@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(summary, label):
    return summary[summary == summary[label]].index


# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image: np.ndarray, boxes: pd.DataFrame, header:str , description:str):
    # Superpose the semi-transparent object detection boxes.    # Colors for the boxes
    LABEL_COLORS = {
        "smoke": [255, 0, 0],
        "fire": [0, 255, 0],
        "cloud": [0, 0, 255],
    }
    image_with_boxes = image.astype(np.float64)
    if isinstance(boxes, pd.DataFrame):
        for _, (xmin, ymin, xmax, ymax, label) in boxes.iterrows():
            image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] += LABEL_COLORS[label]
            image_with_boxes[int(ymin):int(ymax), int(xmin):int(xmax), :] /= 2

    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)
    if GARBAGE_COLLECT:
        del image, image_with_boxes
