import shutil

import numpy as np
import os
import glob


def main(path_to_dataset_dir: str, path_to_output_dir: str, class_name="Smoke",
         split_percent={"train": 0.7, "val": 0.2, "test": 0.1}):
    # collect scenes from dataset dir
    all_data_sources = glob.glob(os.path.join(path_to_dataset_dir, "images", "*"))
    all_scenes = []
    for source_folder in all_data_sources:
        all_scenes += glob.glob(os.path.join(source_folder, "*"))

    # split scenes into train, val, test
    no_training_examples = int(split_percent["train"] * len(all_scenes))
    no_val_examples = int(split_percent["val"] * len(all_scenes))
    no_test_examples = len(all_scenes) - (no_training_examples + no_val_examples)
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

    # find corresponding label file for each image file
    for id, split in {"train": train_scenes, "val": val_scenes, "test": test_scenes}.items():
        for scene in split:
            images = glob.glob(os.path.join(scene, "*"))
            for image in images:
                # copy image to images folder
                shutil.copy(image, os.path.join(path_to_output_dir, id, "images", os.path.split(image)[-1]))

                # copy label to labels folder
                label_filename = os.path.splitext(os.path.basename(image))[0] + ".txt"
                path_to_label = os.path.join(path_to_dataset_dir, "labels", label_filename)
                shutil.copy(path_to_label,
                    os.path.join(path_to_output_dir, id, "labels", os.path.split(path_to_label)[-1]))

    # generate data.yaml
    yaml = '''train: ../train/images
val: ../val/images

nc: 1
names: {class_name}    
    '''.format(class_name=class_name)

    with open(os.path.join(path_to_output_dir, "data.yaml"), "w") as outfile:
        outfile.write(yaml)


if __name__ == '__main__':
    main(path_to_dataset_dir="/tmp/consolidated", path_to_output_dir="/tmp/yolo_dataset")
