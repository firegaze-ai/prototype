import argparse
import shutil
from typing import Dict, Optional, List, Tuple

import numpy as np
import os
import glob
import simplejson as json


def calculate_split_ratios(split_percent: Dict[str, float], all_data: List[str], image_or_scene: str) -> \
        Tuple[int, int, int]:
    no_training_examples = int(split_percent["train"] * len(all_data))
    print("Number of training {}: {}".format(image_or_scene, no_training_examples))
    no_val_examples = int(split_percent["val"] * len(all_data))
    print("Number of validation {}: {}".format(image_or_scene, no_val_examples))
    no_test_examples = len(all_data) - (no_training_examples + no_val_examples)
    print("Number of test {}: {}".format(image_or_scene, no_test_examples))
    return no_training_examples, no_val_examples, no_test_examples


def split_into_subsets(all_data: List[str], no_training_examples: int, no_val_examples: int,
                       no_test_examples: int) -> \
        Tuple[int, int, int]:
    train_scenes = np.random.choice(all_data, size=no_training_examples, replace=False)
    remaining_scenes = [scene for scene in all_data if scene not in train_scenes]
    val_scenes = np.random.choice(remaining_scenes, size=no_val_examples, replace=False)
    test_scenes = [scene for scene in remaining_scenes if scene not in val_scenes]
    return train_scenes, val_scenes, test_scenes


def copy_images_and_labels(key: str, path_to_image: str, path_to_dataset_dir: str, path_to_output_dir: str) -> \
Tuple[str, str]:
    image_filename = os.path.split(path_to_image)[-1]
    shutil.copy(path_to_image, os.path.join(path_to_output_dir, key, "images", image_filename))

    # copy label to labels folder
    label_filename = os.path.splitext(os.path.basename(path_to_image))[0] + ".txt"
    path_to_label = os.path.join(path_to_dataset_dir, "labels", label_filename)
    shutil.copy(path_to_label,
        os.path.join(path_to_output_dir, key, "labels", os.path.split(path_to_label)[-1]))
    return image_filename, label_filename


def main(path_to_dataset_dir: str, path_to_output_dir: str, input_version: str,
         output_version: str, split_by_scene: bool,
         dataset_json: Optional[str],
         split_percent: Dict,
         class_name: Optional[str] = "Smoke"):
    # TODO: enable dataset replication functionality

    print(split_by_scene)

    # collect scenes from dataset dir
    all_data_sources = glob.glob(os.path.join(path_to_dataset_dir, "images", "*"))
    all_scenes = []
    for source_folder in all_data_sources:
        all_scenes += glob.glob(os.path.join(source_folder, "*"))

    if split_by_scene:
        # split scenes into train, val, test
        no_training_examples, no_val_examples, no_test_examples = calculate_split_ratios(split_percent,
            all_scenes, image_or_scene="scenes")

        train_data, val_data, test_data = split_into_subsets(all_scenes, no_training_examples,
            no_val_examples, no_test_examples)

        print(train_data, val_data, test_data)

    else:
        all_images = []
        for datapoint in all_scenes:
            all_images += glob.glob(os.path.join(datapoint, "*"))
        print(all_images)

        # split images into train, val, test
        no_training_examples, no_val_examples, no_test_examples = calculate_split_ratios(split_percent,
            all_images, image_or_scene="images")

        train_data, val_data, test_data = split_into_subsets(all_images, no_training_examples,
            no_val_examples, no_test_examples)

        print(train_data, val_data, test_data)

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
    for key, split in {"train": train_data, "val": val_data, "test": test_data}.items():
        all_images = []
        all_labels = []
        for datapoint in split:
            if split_by_scene:
                images = glob.glob(os.path.join(datapoint, "*"))
                for image in images:
                    image_filename, label_filename = copy_images_and_labels(key, image, path_to_dataset_dir,
                        path_to_output_dir)
                    all_images.append(image_filename)
                    all_labels.append(label_filename)
            else:
                image = datapoint
                image_filename, label_filename = copy_images_and_labels(key, image, path_to_dataset_dir,
                    path_to_output_dir)
                all_images.append(image_filename)
                all_labels.append(label_filename)

        dataset_log.update({key: {"images": all_images, "labels": all_labels}})
    dataset_log.update({"is_split_by_scenes": True if split_by_scene else False})
    dataset_log.update({"dataset_input_version": input_version })
    dataset_log.update({"dataset_output_version": output_version})
    dataset_log.update({"split_ratios": split_percent})

    print("Number of training images: {}\n"
          "Number of validation images: {}\n"
          "Number of test images: {}\n".format(
        len(dataset_log["train"]["images"]),
        len(dataset_log["val"]["images"]),
        len(dataset_log["test"]["images"]),
    )
    )

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
    parser.add_argument('-tr', '--train', help='Train split ratios', type=float, required=True)
    parser.add_argument('-v', '--val', help='Val split ratio', type=float, required=True)
    parser.add_argument('-te', '--test', help='Test split ratio', type=float, required=True)
    # parser.add_argument('-s', '--split_by_scene',
    #     help='Make sure train/val/test do not contain frames from the same scene', type=bool, required=True)
    parser.add_argument('--split_by_scene', dest='split_by_scene', action='store_true')
    parser.add_argument('--no-split_by_scene', dest='split_by_scene', action='store_false')
    parser.set_defaults(feature=True)
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
        split_percent={"train": args["train"], "val": args["val"], "test": args["test"]},
        split_by_scene=args["split_by_scene"],
        dataset_json=None)
