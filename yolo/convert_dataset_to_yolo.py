import argparse
import shutil
from typing import Dict, Optional

import numpy as np
import os
import glob
import simplejson as json


def main(path_to_dataset_dir: str, path_to_output_dir: str, input_version: str,
         output_version: str, dataset_json: Optional[str],
         split_percent: Optional[Dict] = {"train": 0.8, "val": 0.1, "test": 0.1},
         class_name: Optional[str] = "Smoke"):
    # TODO: enable dataset replication functionality

    # collect scenes from dataset dir
    all_data_sources = glob.glob(os.path.join(path_to_dataset_dir, "images", "*"))
    all_scenes = []
    for source_folder in all_data_sources:
        all_scenes += glob.glob(os.path.join(source_folder, "*"))

    # split scenes into train, val, test
    no_training_examples = int(split_percent["train"] * len(all_scenes))
    print("Number of training scenes: {}".format(no_training_examples))
    no_val_examples = int(split_percent["val"] * len(all_scenes))
    print("Number of validation scenes: {}".format(no_val_examples))

    no_test_examples = len(all_scenes) - (no_training_examples + no_val_examples)
    print("Number of test scenes: {}".format(no_test_examples))
    train_scenes = np.random.choice(all_scenes, size=no_training_examples, replace=False)
    remaining_scenes = [scene for scene in all_scenes if scene not in train_scenes]
    val_scenes = np.random.choice(remaining_scenes, size=no_val_examples, replace=False)
    test_scenes = [scene for scene in remaining_scenes if scene not in val_scenes]

    # make directories for train, val, test
    os.makedirs(os.path.join(path_to_output_dir, "train", "images"), exist_ok=True)
    os.makedirs(os.path.join(path_to_output_dir, "train", "labels"), exist_ok=True)
    os.makedirs(os.path.join(path_to_output_dir, "val", "images"), exist_ok=True)
    os.makedirs(os.path.join(path_to_output_dir, "val", "labels"), exist_ok=True)
    os.makedirs(os.path.join(path_to_output_dir, "test", "images"), exist_ok=True)
    os.makedirs(os.path.join(path_to_output_dir, "test", "labels"), exist_ok=True)

    # init log
    dataset_log = {}

    # find corresponding label file for each image file
    for id, split in {"train": train_scenes, "val": val_scenes, "test": test_scenes}.items():
        all_images = []
        all_labels = []
        for scene in split:
            images = glob.glob(os.path.join(scene, "*"))
            for image in images:
                # copy image to images folder
                image_filename = os.path.split(image)[-1]
                shutil.copy(image, os.path.join(path_to_output_dir, id, "images", image_filename))

                # copy label to labels folder
                label_filename = os.path.splitext(os.path.basename(image))[0] + ".txt"
                path_to_label = os.path.join(path_to_dataset_dir, "labels", label_filename)
                shutil.copy(path_to_label,
                    os.path.join(path_to_output_dir, id, "labels", os.path.split(path_to_label)[-1]))
                all_images.append(image_filename)
                all_labels.append(label_filename)

        dataset_log.update({id: {"images": all_images, "labels": all_labels}})
    dataset_log.update({"dataset_version": input_version + "." + output_version})

    with open(os.path.join(path_to_output_dir, "dataset.json"), "w") as outfile:
        json.dump(dataset_log, outfile, indent=4, sort_keys=True)

    # generate data.yaml
    yaml = '''train: ../train/images
val: ../val/images

nc: 1
names: {class_name}    
    '''.format(class_name='["' + class_name + '"]')

    with open(os.path.join(path_to_output_dir, "data.yaml"), "w") as outfile:
        outfile.write(yaml)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Take source dataset in custom format (separated by scenes) and output dataset in yolo format')
    parser.add_argument('-i', '--input', help='Source dataset directory', required=True)
    parser.add_argument('-o', '--output', help='Output dataset directory', required=True)
    parser.add_argument('-iv', '--input_version', help='Input version of the dataset', required=True)
    parser.add_argument('-ov', '--output_version',
        help='Output version of the dataset (new combination of source data)', required=True)

    args = vars(parser.parse_args())

    # main(path_to_dataset_dir="/tmp/consolidated",
    #     path_to_output_dir="/tmp/yolo_dataset",
    #     input_version="1.0",
    #     output_version="1", dataset_json=None)

    main(path_to_dataset_dir=args["input"],
        path_to_output_dir=args["output"],
        input_version=args["input_version"],
        output_version=args["output_version"],
        dataset_json=None)
