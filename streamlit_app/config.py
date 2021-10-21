import os

# Path to the Streamlit public S3 bucket
DATA_URL_ROOT = os.path.join(os.path.dirname(__file__), "data")

# External files to download.
EXTERNAL_DEPENDENCIES = {
    "yolov3.weights": {
        "url": "https://pjreddie.com/media/files/yolov3.weights",
        "size": 248007048
    },
    "yolov3.cfg": {
        "url": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg",
        "size": 8342
    }
}

RESULTS_DIR = os.path.join(DATA_URL_ROOT, "results")
LIVE_IMAGES_DIR = os.path.join(DATA_URL_ROOT, "live_images")
STATIC_IMAGES_DIR = os.path.join(DATA_URL_ROOT, "static_images")
