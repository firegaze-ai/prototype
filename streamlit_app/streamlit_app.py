import gc
import urllib

import streamlit as st
import os
import validators
import matplotlib.pyplot as plt
from memory_profiler import profile

# Streamlit encourages well-structured code, like starting execution in a main() function.
from config import DATA_URL_ROOT, EXTERNAL_DEPENDENCIES, STATIC_IMAGES_DIR
from inference import yolo_v5, batch_parse_yolo_labels_to_csv
from tools import download_file, load_image_from_url, load_image_from_file, load_metadata, create_summary, \
    init_folders
from ui_elements import frame_selector_ui, object_detector_ui, draw_image_with_boxes


def main():
    # Render the readme as markdown using st.markdown.
    with open(os.path.join(os.path.dirname(__file__), "instructions.md"), "r") as infile:
        data = infile.read()
    readme_text = st.markdown(data)

    # Download external dependencies.
    for filename in EXTERNAL_DEPENDENCIES.keys():
        download_file(filename)

    # Once we have the dependencies, add a selector for the app mode on the sidebar.
    st.sidebar.title("What to do")

    app_mode = st.sidebar.selectbox("Choose the app mode",
        ["Show instructions", "Run the app on static example dataset", "Run the app on live cams"
         ])
    if app_mode == "Show instructions":
        st.sidebar.success('To continue select "Run the app on static example dataset".')
    elif app_mode == "Run the app on static example dataset":
        readme_text.empty()
        run_the_app_static()
    elif app_mode == "Run the app on live cams":
        readme_text.empty()
        run_the_app_live()


# This is the main app app itself, which appears when the user selects "Run the app on static dataset".
# @profile
def run_the_app_static():
    init_folders()

    # An amazing property of st.cached functions is that you can pipe them into
    # one another to form a computation DAG (directed acyclic graph). Streamlit
    # recomputes only whatever subset is required to get the right answer!

    path_to_labels_csv = os.path.join(STATIC_IMAGES_DIR, "labels.csv")
    batch_parse_yolo_labels_to_csv(STATIC_IMAGES_DIR, path_to_labels_csv)

    metadata = load_metadata(path_to_labels_csv)
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
    image_url = os.path.join(STATIC_IMAGES_DIR, selected_frame)
    if os.path.isfile(image_url):
        image = load_image_from_file(image_url)
    elif validators.url(image_url):
        image = load_image_from_url(image_url)
    gc.collect()

    # Add boxes for objects on the image. These are the boxes for the ground image.
    boxes = metadata[metadata.frame == selected_frame].drop(columns=["frame"])
    draw_image_with_boxes(image, boxes, "Ground Truth",
        "**Human-annotated data** (frame `%i`)" % selected_frame_index)

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    yolo_boxes = yolo_v5(image_url, image, confidence_threshold, overlap_threshold)
    draw_image_with_boxes(image, yolo_boxes, "Real-time Computer Vision",
        "**YOLO v5 Model** (overlap `%3.1f`) (confidence `%3.1f`)" % (
            overlap_threshold, confidence_threshold))
    del yolo_boxes
    gc.collect()


# This is the main app app itself, which appears when the user selects "Run the app on static dataset".
@profile
def run_the_app_live():
    # avoids the "DuplicateWidgetID" warning
    if "RefreshButton" in st.session_state:
        del st.session_state["RefreshButton"]

    # creates a button and assigns a callback
    if st.button('Refresh', key="RefreshButton"):
        run_the_app_live()

    init_folders()

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
        path_to_image = os.path.join(DATA_URL_ROOT, "live_images", image_name)
        try:
            image = load_image_from_url(image_url)
        except urllib.error.URLError as e:
            print(e)
            image = load_image_from_file(os.path.join(DATA_URL_ROOT, "stream_not_found.png"))
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

    del yolo_boxes, images
    gc.collect()

if __name__ == "__main__":
    main()
