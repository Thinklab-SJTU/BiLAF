import json
import os
import numpy as np
import random
import argparse
import functools
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import utils

torch.autograd.set_detect_anomaly(True)
eps = 1e-10
infty = 1e10


class NCEModel(nn.Module):
    def __init__(self, features, sample_num, temperature, init, distance, balance=1.0):
        super(NCEModel, self).__init__()
        self.features = features
        self.total_num = features.shape[0]
        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance

        self.init = init
        self.distance = distance

        centroids = self.init_centroids()
        self.centroids = nn.Parameter(centroids).cuda()
        print(self.centroids.shape)

    def init_centroids(self):
        if self.init == "random":
            sample_ids = list(range(self.total_num))
            sample_ids = random.sample(sample_ids, self.sample_num)
        elif self.init == "fps":
            dist_func = functools.partial(utils.get_distance, type=self.distance)
            sample_ids = utils.farthest_distance_sample(self.features, self.sample_num, dist_func)
        elif self.init == "kppseed":
            dist_func = functools.partial(utils.get_distance, type=self.distance)
            sample_ids = utils.kmeanspp_seed(self.features, self.sample_num, dist_func)

        centroids = self.features[sample_ids].clone()
        return centroids

    def get_nce_loss(self):
        centroids = F.normalize(self.centroids, dim=1)
        prod = torch.matmul(self.features, centroids.transpose(1, 0))  # (n, k)
        prod = prod / self.temperature
        prod_exp = torch.exp(prod)
        # prod_exp_sum = torch.sum(prod_exp, dim=0)  # (k, )
        prod_exp_pos, pos_k = torch.max(prod_exp, dim=1)  # (n, )
        # prob_pos = prod_exp_pos / prod_exp_sum[pos_k]  # (n, )
        # pos_num = torch.bincount(pos_k, minlength=self.sample_num).float()
        # print(torch.sort(pos_num, descending=False)[0][:100], torch.median(pos_num), torch.std(pos_num))

        cent_prod = torch.matmul(centroids.detach(), centroids.transpose(1, 0))  # (k, k)
        cent_prod = cent_prod / self.temperature
        cent_prod_exp = torch.exp(cent_prod)
        # cent_prob = cent_prod_exp / prod_exp_sum.unsqueeze(0)  # (k, k)
        cent_prob_exp_sum = torch.sum(cent_prod_exp, dim=0)  # (k, )

        mul = prod_exp_pos / (prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        # print(torch.mean(prob_pos), torch.mean(cent_prob))
        J = torch.log(mul)
        J = -torch.mean(J)

        return J


def optimize_dist(features, sample_num, args):
    #  features: (n, c)
    nce_model = NCEModel(features, sample_num, args.temperature, args.init, args.distance, args.balance)
    nce_model = nce_model.cuda()

    optimizer = optim.Adam(nce_model.parameters(), lr=args.lr)
    if args.scheduler != "none":
        if args.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.max_iter, eta_min=1e-6)
        else:
            raise NotImplementedError

    for i in range(args.max_iter):
        nce = nce_model.get_nce_loss()
        loss = nce.item()
        optimizer.zero_grad()
        nce.backward()
        optimizer.step()
        if args.scheduler != "none":
            scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        print("Iter: %d, lr: %.6f, nce: %f"%(i, lr, loss))

        # centroids = nce_model.centroids
        # std = torch.std(centroids, dim=0)
        # std = torch.mean(std).item()
        # print("Iter: %d, lr: %.6f, nce: %f, std: %f"%(i, lr, loss, std))

    centroids = nce_model.centroids.detach()
    centroids = F.normalize(centroids, dim=1)
    dist = torch.matmul(centroids, features.transpose(1, 0))  # (k, n)
    _, sample_ids = torch.max(dist, dim=1)
    sample_ids = sample_ids.cpu().numpy().tolist()
    print(len(sample_ids), len(set(sample_ids)))
    sample_ids = set()
    _, ids_sort = torch.sort(dist, dim=1, descending=True)
    for i in range(ids_sort.shape[0]):
        for j in range(ids_sort.shape[1]):
            if ids_sort[i, j].item() not in sample_ids:
                sample_ids.add(ids_sort[i, j].item())
                break
    print(len(sample_ids))
    sample_ids = list(sample_ids)
    return sample_ids


def merge_features(all_features):
    merged_features = torch.Tensor(np.array(list(all_features.values())))
    # merged_features = [torch.from_numpy(feature) for feature in merged_features]
    # merged_features = torch.cat(merged_features, dim=0)
    # print(len(merged_features))
    # merged_features = [torch.Tensor(x) for x in merged_features]
    # print(merged_features[0].shape)

    id2idx = {}
    idx = 0
    count = 0
    merged_ids = []
    for id in all_features:
        id2idx[count] = torch.arange(idx, idx + all_features[id].shape[0])
        merged_ids.append(id)
        idx = idx + all_features[id].shape[0]
        count += 1
    return merged_features, merged_ids, id2idx

def main(args):
    import pickle
    with open(args.feature_path, "rb") as file:
        all_features = pickle.load(file)
    features, ids_name, id2idx = merge_features(all_features)
    features = torch.Tensor(features).cuda()

    print(features.shape)

    total_num = features.shape[0]
    # In PascalVOC, the sample_num is decied directly by the args.sample_nums
    # sample_num = int(total_num * args.percent * 0.01)
    sample_num = args.sample_nums

    print("Total num: %d, sample num: %d" % (total_num, sample_num))

    features = F.normalize(features, dim=1)

    sample_ids = optimize_dist(features, sample_num, args)
    sample_ids.sort()

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

    filename_07 = f"PascalVoc_ActiveFT_trainval_iter_{args.max_iter}_sampleNum_{sample_num}_VOC2007.txt"
    filename_12 = f"PascalVoc_ActiveFT_trainval_iter_{args.max_iter}_sampleNum_{sample_num}_VOC2012.txt"

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    with open(os.path.join(args.output_dir, filename_07), "w") as file:
        file.writelines(samples_07)
    with open(os.path.join(args.output_dir, filename_12), "w") as file:
        file.writelines(samples_12)



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize extracted features')

    parser.add_argument('--feature_path',  default='features/PascalVoc/PascalVoc_dino_vit_small_features.pkl',
                        type=str, help='path of saved features')

    parser.add_argument('--output_dir', default='features/PascalVoc', type=str, help='dir to save the visualization')
    parser.add_argument('--filename', default=None, type=str, help='filename of the visualization')
    parser.add_argument('--temperature', default=0.07, type=float, help='temperature for softmax')
    parser.add_argument('--threshold', default=0.0001, type=float, help='convergence threshold')
    parser.add_argument('--max_iter', default=300, type=int, help='max iterations')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--percent', default=10, type=float, help='sample percent')
    parser.add_argument('--sample_nums', default=1000, type=int, help='sample percent')
    parser.add_argument('--init', default='random', type=str, choices=['random', 'fps', 'kppseed'])
    parser.add_argument('--distance', default='euclidean', type=str, help='euclidean or cosine')
    parser.add_argument('--scheduler', default='none', type=str, help='scheduler')
    parser.add_argument('--balance', default=1.0, type=float, help='balance ratio')

    parser.add_argument('--save_name', type=str, default="dino_vits16_ActiveFT")
    args = parser.parse_args()
    main(args)