import numpy as np
import torch
import os
import json
import argparse
import random
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

def main(args):
    input = np.load(args.feature_path)
    features, _ = input[:, :-1], input[:, -1]

    total_num = features.shape[0]
    sample_num = int(total_num * args.percent * 0.01)

    if args.filename is None:
        name = args.feature_path.split("/")[-1]
        name = name[:-4]
        args.filename = f"{name}_{args.strategy}_sampleNum_{sample_num}.json"

    # args.output_dir = os.path.join(args.output_dir, args.strategy)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    output_path = os.path.join(args.output_dir, args.filename)

    sample_ids = sample(features, sample_num, args.strategy)

    with open(output_path, "w") as file:
        json.dump(sample_ids, file)

def sample(features, sample_num, strategy):
    if strategy == "Random":
        return random.sample(range(len(features)), sample_num)
    elif strategy == "FDS":
        return fds_sampling(features, sample_num)
    elif strategy == "Kmeans":
        return kmeans_sampling(features, sample_num)
    else:
        raise ValueError("Unknown sampling strategy")

def fds_sampling(features, sample_num):
    selected_ids = [random.randint(0, len(features) - 1)]
    for i in range(1, sample_num):
        print(i, sample_num)
        distances = cdist(features, features[selected_ids], 'euclidean').min(axis=1)
        selected_ids.append(int(np.argmax(distances)))
    return selected_ids

def kmeans_sampling(features, sample_num):
    kmeans = KMeans(n_clusters=sample_num, random_state=0).fit(features)
    centers = kmeans.cluster_centers_

    distance_matrix = cdist(centers, features)
    closest_indices = np.argmin(distance_matrix, axis=1)

    print(closest_indices.shape)
    print(closest_indices)

    return closest_indices.tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('--feature_path', default='../features/CIFAR10_train.npy', type=str, help='path of saved features')
    parser.add_argument('--output_dir', default='../features/Baselines', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('--percent', default=1, type=float, help='sample percent')
    parser.add_argument('--strategy', default=None, type=str, choices=["Random", "FDS", "Kmeans"], help='sampling strategy')
    args = parser.parse_args()

    args.strategy = "Kmeans"

    main(args)
