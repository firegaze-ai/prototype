import copy
import datetime
import glob
import os
import shutil
import time
import dill
from typing import List, Tuple

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
import simplejson as json


class ClusteringConfig:
    def __init__(self, image_resize_shape: Tuple[int, int], cluster_eps: float, cluster_min_samples: int,
                 path_to_input_dir: str, path_to_output_dir: str, ):
        self.image_resize_shape = image_resize_shape
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.path_to_input_dir = path_to_input_dir
        self.path_to_output_dir = path_to_output_dir
        self._distance_matrix: np.ndarray = None
        self.paths_to_images = self.get_images()

    @property
    def distance_matrix(self):
        return self._distance_matrix

    @distance_matrix.setter
    def distance_matrix(self, matrix):
        self._distance_matrix = matrix

    def load_distance_matrix(self, path_to_file: str):
        with open(path_to_file, "rb") as infile:
            self._distance_matrix = dill.load(infile)

    def dump_distance_matrix(self, path_to_file: str):
        with open(path_to_file, "wb") as outfile:
            dill.dump(self._distance_matrix, outfile)

    def get_images(self):
        return glob.glob(os.path.join(self.path_to_input_dir, "*.jpeg")) + glob.glob(
            os.path.join(self.path_to_input_dir, "*.jpg"))

    def to_disk(self, path_to_file: str):
        with open(path_to_file, "w") as outfile:
            obj = copy.copy(self.__dict__)
            obj.pop("_distance_matrix")
            json.dump(obj, outfile)

    def from_disk(self, path_to_file: str):
        with open(path_to_file, "r") as infile:
            self.__dict__.update(json.load(infile))


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


def compute_mds(distance_matrix: np.ndarray, clustering_config: ClusteringConfig):
    embedding = MDS(n_components=2, dissimilarity='precomputed')
    X_transformed = embedding.fit_transform(distance_matrix[:distance_matrix.shape[0]])
    X_transformed.shape
    return X_transformed


def cluster(mds_result, clustering_config: ClusteringConfig):
    # Compute DBSCAN
    db = DBSCAN(
        eps=clustering_config.cluster_eps,
        min_samples=clustering_config.cluster_min_samples,
        metric="precomputed"
    ).fit(
        clustering_config.distance_matrix)
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


def compute_distance_matrix_parallelized(paths_to_images: str, clustering_config: ClusteringConfig):
    preloaded_images = [
        np.array(Image.open(path).convert('L').resize(clustering_config.image_resize_shape, Image.ANTIALIAS))
        for path
        in paths_to_images]
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

    # save into config class
    clustering_config.distance_matrix = W
    clustering_config.dump_distance_matrix(path_to_file=os.path.join(
        clustering_config.path_to_input_dir, "distance_matrix.dill"))

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
    is_distance_matrix_caching_enabled: bool = True
    path_to_input_dir: str = "/home/orphefs/Downloads/fires_smoke_forest_dataset/aiformankind/day_time_wildfire_v2/images/clustered_210917125953/-1"

    clustering_config = ClusteringConfig(
        image_resize_shape=(128, 128),
        cluster_eps=0.4,
        cluster_min_samples=8,
        path_to_input_dir=path_to_input_dir,
        path_to_output_dir=os.path.join(path_to_input_dir,
            "clustered_" + datetime.datetime.now().strftime("%y%m%d%H%M%S")))

    if is_distance_matrix_caching_enabled:
        clustering_config.load_distance_matrix(path_to_file=os.path.join(
            clustering_config.path_to_input_dir, "distance_matrix.dill"))
    else:
        clustering_config.distance_matrix = compute_distance_matrix_parallelized(
            clustering_config.paths_to_images,
            clustering_config)

    mds_result = compute_mds(clustering_config.distance_matrix, clustering_config)
    plt.scatter(mds_result[:, 0], mds_result[:, 1])
    cluster_labels = cluster(mds_result, clustering_config)
    copy_to_folder(clustering_config.paths_to_images,
        cluster_labels, clustering_config.path_to_output_dir)

    plt.show()

    clustering_config.to_disk(os.path.join(clustering_config.path_to_input_dir, "clustering_config.json"))

if __name__ == '__main__':
    main()
