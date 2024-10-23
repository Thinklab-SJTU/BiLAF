import copy
import os
import numpy as np
import random
import argparse
import functools
import torch
from scipy.spatial.distance import cdist
import time
import pickle
import json

eps = 1e-10
infty = 1e10


def distribute_budget(budget, proportions):
    proportions = np.array(proportions)
    proportions = 1.0 * proportions / proportions.sum()

    allocations = np.floor(proportions * budget).astype(int)

    remaining_budget = budget - allocations.sum()

    while remaining_budget > 0:
        cur_index = 0
        for i in range(len(allocations)):
            if ((allocations[i] < allocations[cur_index]) or
                    (allocations[i] == allocations[cur_index] and proportions[i] > proportions[cur_index])):
                cur_index = i
        allocations[cur_index] += 1
        remaining_budget -= 1

    return allocations


"""
    Find the class index for each sample
    Return class2index_list[0...nums-1], class2index_list[i] is the list of indices for class i
"""


def distance_cluster(features, class_core_features, nums):
    dist_matrix = cdist(features, class_core_features, 'euclidean')
    min_indices = np.argmin(dist_matrix, axis=1)
    print(len(min_indices), len(set(min_indices)))
    class2index_list = [[] for _ in range(nums)]
    for i in range(len(min_indices)):
        class2index_list[min_indices[i]].append(i)
    return class2index_list


"""
    Step by Step Density Cluster and remove the furthest pc% samples
"""


def density_cluster_denoise(class2index_list, features, indices, nums,
                            density_nearst_number,
                            remove_percentage,
                            cycle_percentage):
    cycle_levels = np.zeros(len(features))
    for class_idx in range(nums):
        # class2index_list[class_idx][i] is the index of the i-th sample in the original dataset
        cur_features = features[class2index_list[class_idx]]
        if len(cur_features) < 10:
            continue
        print(class_idx, len(cur_features))

        cur_cycle_levels = np.zeros(len(cur_features))
        core_idx = indices[class_idx]
        cor_cur_idx = -1
        for i in range(len(cur_features)):
            if core_idx == class2index_list[class_idx][i]:
                cor_cur_idx = i
                break
        assert cor_cur_idx != -1

        cur_dist_matrix = cdist(cur_features, cur_features, 'euclidean')
        cycle_number = int(len(cur_features) * cycle_percentage)
        if cycle_number == 0:
            cycle_number = 1
        # 1. Initialize the nearest cycle_percentage% samples of the core as the density center
        arr = [(cur_dist_matrix[cor_cur_idx][i], i) for i in range(len(cur_dist_matrix[cor_cur_idx]))]
        arr.sort()
        cluster = [x[1] for x in arr[:cycle_number]]
        cur_cycle_levels[cluster] = 1

        # 2. Step by Step construct the density center, each step we add the nearest cycle_percentage% samples
        for itrs in range(2, int(len(cur_features) / cycle_number) + 1):
            # Find the unselected samples,
            candidate = [i for i in range(len(cur_cycle_levels)) if cur_cycle_levels[i] == 0]
            if len(candidate) == 0:
                break

            # Calculate the mean distance to the nearest density_nearst_number samples in the cluster
            distance = []
            for i in candidate:
                nearest_distance = np.argsort(cur_dist_matrix[i][cluster])[:density_nearst_number]
                distance.append((np.mean(cur_dist_matrix[i][nearest_distance]), i))

            # Select the sample with the smallest mean distance
            distance.sort()
            new_cluster = [x[1] for x in distance[:cycle_number]]
            cur_cycle_levels[new_cluster] = itrs
            cluster.extend(new_cluster)

        # Update the cycle_levels for each sample
        cycle_levels[class2index_list[class_idx]] = cur_cycle_levels

        # 3. Remove the furthest pc% samples
        cluster = cluster[:int(len(cluster) * (1 - remove_percentage))]
        class2index_list[class_idx] = [class2index_list[class_idx][x] for x in cluster]

    return class2index_list, cycle_levels


"""
For each class we select the given budget samples
    Step2: Calculate the boundary score for samples in each class.
           Obtain the rings level of each sample.
    Step3: Select the points step by step following the rules

    Parameters:
    - same_oppo_penalty_ratio: The penalty ratio increase for outer distance of the same opponent class
                                 to avoid all boundary points are from the same opponent class
        Therefore, the boundary score is calculated as: 
            (outer_distance * same_oppo_penalty_ratio^n - inner_distance) / max(outer_distance, inner_distance)
"""


def selection(class2index_list, class_core_features, features, indices, nums, budget,
              same_oppo_penalty_ratio):
    boundary_points = []
    proportions = [len(index_list) for index_list in class2index_list]
    allocations = distribute_budget(budget, proportions)

    for class_idx in range(nums):
        """ Step2: Calculate the boundary score for samples in each class. """
        cur_class_nums = len(class2index_list[class_idx])
        cur_class_indices = class2index_list[class_idx]

        core_index = -1

        for i in range(cur_class_nums):
            if cur_class_indices[i] == indices[class_idx]:
                core_index = i
                break

        if allocations[class_idx] == 1:
            boundary_points.append(int(cur_class_indices[core_index]))
            print(class_idx, len(cur_class_indices))
            continue

        cur_features = features[cur_class_indices]

        inner_dist_matrix = cdist(cur_features, cur_features, 'euclidean')
        inner_scores = []
        for i, distances in enumerate(inner_dist_matrix):
            inner_distance = np.mean(distances)
            inner_scores.append(inner_distance)

        outer_dist_matrix = cdist(cur_features, class_core_features, 'euclidean')
        outer_scores = []
        outer_oppo_class = []
        for i in range(cur_class_nums):
            outer_dist_matrix[i][class_idx] = 1e9 + 7
            outer_scores.append(np.min(outer_dist_matrix[i]))
            outer_oppo_class.append(np.argmin(outer_dist_matrix[i]))

        class_penalty_ratio = [1.0 for _ in range(nums)]

        """
            Step3: Select the points step by step
                Upon Selecting the point, we will remove some of the nearst ones in the candidate set
        """


        # class_budget = budget // nums
        # class_budget = 1
        class_budget = allocations[class_idx]
        cur_class_list = [core_index]
        invalid_number_per_node = int(cur_class_nums / class_budget)
        candidates = [i for i in range(cur_class_nums)]

        while len(cur_class_list) < class_budget:
            # Set #invalid_number nearset points invalid and remove them in the candidate set
            node = cur_class_list[-1]
            valid = []
            for i in candidates:
                valid.append((inner_dist_matrix[node][i], i))
            valid.sort()
            valid = valid[invalid_number_per_node:]
            candidates = [x[1] for x in valid]

            # Update the boundary score for each candidate
            if node != core_index:
                class_penalty_ratio[outer_oppo_class[node]] *= same_oppo_penalty_ratio

            boundary_scores = []
            for i in candidates:
                penalty = class_penalty_ratio[outer_oppo_class[i]]
                boundary_score = ((outer_scores[i] * penalty - inner_scores[i]) /
                                  max(outer_scores[i], inner_scores[i]))
                boundary_scores.append((boundary_score, i))

            # Find the lowest boundary score index in the remaining candidate
            lowest_boundary_score_index = np.argmin([x[0] for x in boundary_scores])
            new_node = boundary_scores[lowest_boundary_score_index][1]

            cur_class_list.append(new_node)

        cur_class_boundary_points = [int(cur_class_indices[index]) for index in cur_class_list]
        boundary_points.extend(cur_class_boundary_points)
        print(class_idx, len(cur_class_indices), class_budget, cur_class_boundary_points)

    print(boundary_points)
    print(len(boundary_points))
    assert np.all(np.isin(indices, boundary_points))

    return boundary_points

def main(args):
    # Initialize the parameters and load the parameters
    cur_number = args.cur_number
    nums = cur_number
    budget = args.budget

    # Separating features and labels
    inputs = np.load(args.features_inputs)
    features = inputs[:, :-1]  # All rows, all but last column
    labels = inputs[:, -1]  # All rows, last column

    indices_file_name = args.indices_file_name
    print(indices_file_name)
    indices = np.array(json.load(open(indices_file_name, 'r')))
    assert len(indices) == nums


    # Step0: Find the class index for each sample
    class_core_features = features[indices]
    class2index_list = distance_cluster(features, class_core_features, nums)

    # Step1: Step by Step Density Cluster and remove the furthest pc% samples
    class2index_list, cycle_levels = density_cluster_denoise(
        class2index_list, features, indices, nums,
        density_nearst_number=args.density_nearst_number,
        remove_percentage=args.remove_percentage,
        cycle_percentage=args.cycle_percentage
    )

    # Step2/3: Find the boundary points
    boundary_points = selection(class2index_list, class_core_features, features, indices, nums, budget,
                                same_oppo_penalty_ratio=args.same_oppo_penalty_ratio)

    debug = False
    if debug:
        with open("boundary.json", 'w') as f:
            json.dump(boundary_points, f)
    else:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        file_name = f"{args.output_dir}/Density_{budget}_from_ActiveFT{cur_number}.json"
        with open(file_name, "w") as f:
            json.dump(boundary_points, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('--feature_path', default='../features', type=str)
    parser.add_argument('--dataset', default='ImageNet', type=str, help='CIFAR10 or CIFAR100')

    parser.add_argument('--init_indices', default=None, type=str, help='path of saved features')
    parser.add_argument('--features_inputs', default=None, type=str)
    parser.add_argument("--cur_number", default=6405, type=int, help="Number of selected samples")
    parser.add_argument("--budget", default=12811, type=int, help="Number of selected samples")
    # 0.25% = 3202 0.5% = 6405 1% = 12811 2% = 25623 5% = 64058 10% = 128116 20% = 256233
    parser.add_argument('--distance', default='euclidean', type=str, help='euclidean or cosine')

    parser.add_argument('--density_nearst_number', default=10, type=int)
    parser.add_argument('--remove_percentage', default=0.2, type=float)
    parser.add_argument('--cycle_percentage', default=0.1, type=float)
    parser.add_argument('--same_oppo_penalty_ratio', default=1.1, type=float)

    args = parser.parse_args()

    args.features_inputs = f"{args.feature_path}/{args.dataset}_train.npy" if args.features_inputs is None else args.features_inputs
    args.indices_file_name = (f"{args.feature_path}/ImageNet/ActiveFT/{args.dataset}_train_ActiveFT_euclidean_temp_0.07_lr_0.001000_scheduler_"
                              f"none_iter_100_sampleNum_{args.cur_number}.json") if args.init_indices is None else args.init_indices
    args.output_dir = f"{args.feature_path}/{args.dataset}/Density_Cluster/" if args.output_dir is None else args.output_dir

    main(args)

