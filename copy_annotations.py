import glob
import os


def copy_to_folder(paths_to_images: List, cluster_labels: np.ndarray, path_to_output_dir: str):
    d = dict(zip(paths_to_images, cluster_labels))

    for path_to_image, cluster in d.items():
        directory = os.path.join(path_to_output_dir, str(cluster))
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copyfile(path_to_image, os.path.join(directory,
            os.path.split(path_to_image)[-1]))

def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)


def map_folder_structure(path_to_folder: str):


def main():
    path_to_root = "/home/orphefs/Downloads/fires_smoke_forest_dataset/aiformankind/day_time_wildfire_v2/annotations/xmls"
    paths_to_xmls = glob.glob(os.path.join(path_to_root,"*.xml"))
    for path_to_xml in paths_to_xmls:
        filename = os.path.split(path_to_xml)[-1]
        find(os.path.basename(filename) + ".jpeg", )



if __name__ == '__main__':
    main()