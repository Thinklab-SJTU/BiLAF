# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from datasets.custom_dataset import ADEReturnIndexDataset, VOCReturnIndexDataset


def extract_features(args):
    # utils.init_distributed_mode(args)
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    if args.framework == "dino" or args.pretrained_weights == "":
        utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    else:
        ckpt_dict = torch.load(args.pretrained_weights)
        if "model" in ckpt_dict:
            ckpt_dict = ckpt_dict["model"]
        model.load_state_dict(ckpt_dict, strict=False)
    # utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built.")

    # ============ preparing data ... ============
    data_transform = pth_transforms.Compose([
        pth_transforms.Resize((224, 224), interpolation=3),
        # pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    all_features = {}
    for year in ["2007", "2012"]:
        dataset_train = VOCReturnIndexDataset(args.data_root, year, transform=data_transform)
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
        print(f"Data loaded with {len(dataset_train)} trainings.")

        # ============ extract features ... ============
        print("Extracting features for train set...")
        train_features, train_ids = validate_network(data_loader_train, model, args.n_last_blocks,
                                                     args.avgpool_patchtokens, args.maxpool_patchtokens)

        for i in range(len(train_features)):
            all_features[train_ids[i]] = train_features[i]

    # outputs = np.concatenate([features, labels], axis=1)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    import pickle
    if args.avgpool_patchtokens:
        pickle_file_path = os.path.join(args.output_dir, "PascalVoc_%s_%s_avgpool_features.pkl"%(args.framework, args.arch))
        with open(pickle_file_path, "wb") as file:
            pickle.dump(all_features, file)
    elif args.maxpool_patchtokens:
        pickle_file_path = os.path.join(args.output_dir, "PascalVoc_%s_%s_maxpool_features.pkl"%(args.framework, args.arch))
        with open(pickle_file_path, "wb") as file:
            pickle.dump(all_features, file)
    else:
        pickle_file_path = os.path.join(args.output_dir, "PascalVoc_%s_%s_features.pkl"%(args.framework, args.arch))
        with open(pickle_file_path, "wb") as file:
            pickle.dump(all_features, file)

@torch.no_grad()
def validate_network(data_loader, model, n, avgpool, maxpool):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    features = None
    targets = None
    for inp, idx in metric_logger.log_every(data_loader, 20, header):
        # move to gpu
        inp = inp.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
                elif maxpool:
                    output = torch.cat((output.unsqueeze(-1), torch.max(intermediate_output[-1][:, 1:], dim=1)[0].unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)

        output = output.cpu().numpy()

        if features is None:
            features = output
            ids = idx
        else:
            features = np.concatenate([features, output], axis=0)
            ids.extend(idx)
    return features, ids


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=1, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--maxpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global max pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--framework', default='dino', type=str, help='framework of pretraining')
    # parser.add_argument('--framework', default='', type=str, help='framework of pretraining')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    # parser.add_argument('--pretrained_weights', default='pretrain/dino_resnet50_pretrain.pth', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--output_dir', default="features/PascalVoc", help='Path to save extracted features')
    parser.add_argument('--data_root', default="data", type=str, help="root dir of Pascal VOC")

    args = parser.parse_args()
    extract_features(args)
