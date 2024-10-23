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
    # print(len(min_indices), len(set(min_indices)))
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

        # print(class_idx, len(cur_features))
        if len(cur_features) < 10:
            continue

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
        core_index = -1
        for i in range(cur_class_nums):
            if cur_class_indices[i] == indices[class_idx]:
                core_index = i
                break

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

        # print(cur_class_list)

        cur_class_boundary_points = [int(cur_class_indices[index]) for index in cur_class_list]
        boundary_points.extend(cur_class_boundary_points)
        # print(len(cur_class_indices), class_budget, cur_class_boundary_points)

    # print(boundary_points)

    assert np.all(np.isin(indices, boundary_points))

    return boundary_points


def merge_features(all_features):
    merged_features = torch.Tensor(np.array(list(all_features.values())))

    id2idx = {}
    idx = 0
    count = 0
    ids_name = []
    for id_str in all_features:
        id2idx[count] = torch.arange(idx, idx + all_features[id_str].shape[0])
        ids_name.append(id_str)
        idx = idx + all_features[id_str].shape[0]
        count += 1
    return merged_features, ids_name, id2idx

def main(args):
    begin_time = time.time()

    import pickle
    with open(args.feature_path, "rb") as file:
        all_features = pickle.load(file)
    features, ids_name, id2idx = merge_features(all_features)

    total_num = features.shape[0]
    cur_number = args.cur_sample_num
    nums = cur_number
    budget = args.budget_sample_num

    filename_07_ActiveFT = f"../features/PascalVoc/PascalVoc_ActiveFT_trainval_iter_300_sampleNum_{cur_number}_VOC2007.txt"
    filename_12_ActiveFT = f"../features/PascalVoc/PascalVoc_ActiveFT_trainval_iter_300_sampleNum_{cur_number}_VOC2012.txt"
    indices = []

    with open(filename_07_ActiveFT, 'r') as f:
        samples_07 = f.readlines()
        for sample in samples_07:
            indices.append(ids_name.index(sample.strip()))

    with open(filename_12_ActiveFT, 'r') as f:
        samples_12 = f.readlines()
        for sample in samples_12:
            indices.append(ids_name.index(sample.strip()))

    print(len(features))
    assert len(indices) == cur_number

    load_time = time.time()
    print(f"Load time: {load_time - begin_time:.4f}")

    # Step0: Find the class index for each sample
    class_core_features = features[indices]
    class2index_list = distance_cluster(features, class_core_features, nums)

    step0_time = time.time()
    print(f"Step0 time: {step0_time - load_time:.4f}")

    # Step1: Step by Step Density Cluster and remove the furthest pc% samples
    class2index_list, cycle_levels = density_cluster_denoise(
        class2index_list, features, indices, nums,
        density_nearst_number=args.density_nearst_number,
        remove_percentage=args.remove_percentage,
        cycle_percentage=args.cycle_percentage
    )

    step1_time = time.time()
    print(f"Step1 time: {step1_time - step0_time:.4f}")

    # Step2/3: Find the boundary points
    sample_ids = selection(class2index_list, class_core_features, features, indices, nums, budget,
                                same_oppo_penalty_ratio=args.same_oppo_penalty_ratio)

    step2_time = time.time()
    print(f"Step2 time: {step2_time - step1_time:.4f}")

    end_time = time.time() - begin_time
    print(f"Total time: {end_time:.4f}")

    sample_ids.sort()
    print("Selected samples: ", len(sample_ids))

    samples_07 = []
    samples_12 = []
    for idx in sample_ids:
        cur_id = ids_name[idx]
        if "_" in cur_id:
            samples_12.append(cur_id + "\n")
        else:
            samples_07.append(cur_id + "\n")

    samples_07.sort()
    samples_12.sort()

    filename_07 = f"PascalVoc_Density_{budget}_from_ActiveFT_{cur_number}_VOC2007.txt"
    filename_12 = f"PascalVoc_Density_{budget}_from_ActiveFT_{cur_number}_VOC2012.txt"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, filename_07), "w") as file:
        file.writelines(samples_07)
    with open(os.path.join(args.output_dir, filename_12), "w") as file:
        file.writelines(samples_12)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')
    parser.add_argument('--feature_path', default='../features/PascalVoc/PascalVoc_dino_vit_small_features.pkl',
                        type=str, help='path of saved features')
    parser.add_argument('--dataset', default='PascalVoc', type=str, help='CIFAR10 or CIFAR100')

    parser.add_argument('--indices_file_name', default=None, type=str)
    parser.add_argument('--output_dir', default='../features/PascalVoc/Density_Cluster', type=str)

    parser.add_argument('--init_indices', default=None, type=str, help='path of saved features')
    parser.add_argument("--cur_sample_num", default=500, type=int, help="Number of selected samples")
    parser.add_argument("--budget_sample_num", default=5000, type=int, help="Number of selected samples")

    parser.add_argument('--distance', default='euclidean', type=str, help='euclidean or cosine')

    parser.add_argument('--density_nearst_number', default=10, type=int)
    parser.add_argument('--remove_percentage', default=0.1, type=float)
    parser.add_argument('--cycle_percentage', default=0.1, type=float)
    parser.add_argument('--same_oppo_penalty_ratio', default=1.1, type=float)

    args = parser.parse_args()

    main(args)
