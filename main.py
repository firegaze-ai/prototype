import glob
import os
import shutil
import time
from typing import List

from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.manifold import MDS
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import hausdorff_distance
from mpire import WorkerPool
from image_similarity_measures import quality_metrics

import matplotlib.pyplot as plt
import numpy as np
import itertools


def compute_histogram_rms_distance(img1, img2):
    hist1, _ = np.histogram(np.reshape(img1, -1), 150, density=True)
    hist2, _ = np.histogram(np.reshape(img2, -1), 150, density=True)

    rms_err = 1 / len(hist1) * np.sum((hist1 - hist2) ** 2)

    # print(rms_err)
    # plt.show()

    return rms_err


def compute_structural_dissimilarity(preloaded_image_combinations, i):
    img1 = preloaded_image_combinations[i][0]
    img2 = preloaded_image_combinations[i][1]
    dissimilarity = 1 - ssim(img1, img2,
        data_range=img2.max() - img2.min())

    return dissimilarity


def compute_haussdorf_distance(preloaded_image_combinations, i):
    img1 = preloaded_image_combinations[i][0]
    img2 = preloaded_image_combinations[i][1]
    dissimilarity = hausdorff_distance(img1, img2)

    return dissimilarity


def compute_mds(distance_matrix):
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(distance_matrix[:distance_matrix.shape[0]])
    X_transformed.shape
    return X_transformed


def cluster(mds_result):
    # Compute DBSCAN
    db = DBSCAN(eps=0.05, min_samples=3).fit(mds_result)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    return labels


def plot_clusters():
    pass


def compute_distance_matrix_parallelized(paths_to_images: str):
    preloaded_images = [np.array(Image.open(path).convert('L').resize((128,128), Image.ANTIALIAS)) for path in paths_to_images]
    image_path_combinations = itertools.combinations(paths_to_images, 2)
    preloaded_image_combinations = list(itertools.combinations(preloaded_images, 2))
    distance_matrix = np.zeros([len(paths_to_images), len(paths_to_images)])
    indices = dict(zip(paths_to_images, [i for i in range(len(paths_to_images))]))

    with WorkerPool(n_jobs=10, shared_objects=preloaded_image_combinations, start_method='fork') as pool:
        results = pool.map(compute_structural_dissimilarity, range(len(preloaded_image_combinations)),
            progress_bar=True)

    k = 0
    for image_pair in image_path_combinations:
        i = indices[image_pair[0]]
        j = indices[image_pair[1]]
        distance_matrix[i, j] = results[k]
        k += 1

    # symmetrize matrix
    W = np.triu(distance_matrix) + np.tril(distance_matrix.T, 1)
    print(W)
    return W


def compute_distance_matrix(paths_to_images: str):
    image_combinations = itertools.combinations(paths_to_images, 2)
    distance_matrix = np.zeros([len(paths_to_images), len(paths_to_images)])
    indices = dict(zip(paths_to_images, [i for i in range(len(paths_to_images))]))
    preloaded_images = [np.array(Image.open(path).convert('L')) for path in paths_to_images]
    for image_pair in image_combinations:
        i = indices[image_pair[0]]
        j = indices[image_pair[1]]
        distance_matrix[i, j] = compute_structural_dissimilarity(
            preloaded_images[i],
            preloaded_images[j])

    # symmetrize matrix
    W = np.triu(distance_matrix) + np.tril(distance_matrix.T, 1)
    print(W)
    return W


def copy_to_folder(paths_to_images: List, cluster_labels: np.ndarray, path_to_output_dir: str):
    d = dict(zip(paths_to_images, cluster_labels))

    for path_to_image, cluster in d.items():
        directory = os.path.join(path_to_output_dir, str(cluster))
        if not os.path.exists(directory):
            os.makedirs(directory)
        shutil.copyfile(path_to_image, os.path.join(directory,
            os.path.split(path_to_image)[-1]))


def benchmark():
    # time just one pair of images
    img1 = np.array(Image.open("/tmp/images/ckagz65fcanl60b12682yukgz.jpeg").convert('L'))
    img2 = np.array(Image.open("/tmp/images/ckagz777enrb20700g9y39sr6.jpeg").convert('L'))


    times = []
    for i in range(1000):
        t0 = time.time()
        dissimilarity = 1 - ssim(img1, img2,
            data_range=img2.max() - img2.min())
        t1 = time.time()
        total = t1 - t0
        times.append(total)

    plt.hist(times, 100)
    plt.show()


def main():
    path_to_dir = "/home/orphefs/Downloads/fires_smoke_forest_dataset/aiformankind/day_time_wildfire_v2/images"
    path_to_output_dir = "/home/orphefs/Downloads/fires_smoke_forest_dataset/aiformankind/day_time_wildfire_v2/images/organized"

    # path_to_dir = "/tmp/images"
    # path_to_output_dir = "/tmp/images/organized"
    paths_to_images = glob.glob(os.path.join(path_to_dir, "*.jpeg"))

    distance_matrix = compute_distance_matrix_parallelized(paths_to_images)

    mds_result = compute_mds(distance_matrix)
    plt.scatter(mds_result[:, 0], mds_result[:, 1])
    cluster_labels = cluster(mds_result)
    copy_to_folder(paths_to_images, cluster_labels, path_to_output_dir)

    plt.show()

    # benchmark()


if __name__ == '__main__':
    main()
