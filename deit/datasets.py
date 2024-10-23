# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset
import pandas as pd

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

class ImageNet_Generate_Dataset(ImageFolder):
    def __init__(self, root, transform=None, subset_ids=None, target_transform=None, loader=default_loader):
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.subset_ids = subset_ids

        targets_file = os.path.join(root, 'labels.json')
        with open(targets_file) as json_file:
            targets = json.load(json_file)

        self.samples = []
        for idx in subset_ids:
            img_name = str(idx) + '.JPEG'
            path_current = os.path.join(root, img_name)
            target_current = targets[img_name]
            self.samples.append((path_current, target_current))

    # __getitem__ and __len__ inherited from ImageFolder
class Cub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, loader=default_loader, download=True):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.loader = default_loader
        self.train = train

        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'IMNETSUBSET_Generate' and not is_train:
        args.data_set = 'IMNETSUBSET'


    if args.data_set == 'CIFAR100':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
    elif args.data_set == 'CIFAR10':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
    elif args.data_set == 'IMNET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == "CUB2011":
        dataset = Cub2011(root=args.data_path, train=is_train, transform=transform)
        nb_classes = 200
    elif args.data_set == 'CUB2011SUBSET':
        dataset = Cub2011(root=args.data_path, train=is_train, transform=transform)
        nb_classes = 200
        if is_train:
            with open(args.subset_ids, 'r') as file:
                ids = json.load(file)
            dataset = torch.utils.data.Subset(dataset, ids)
        print(len(dataset), "\n\n\n")
    elif args.data_set == 'CIFAR10SUBSET':
        dataset = datasets.CIFAR10(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 10
        if is_train:
            with open(args.subset_ids, 'r') as file:
                ids = json.load(file)
            dataset = torch.utils.data.Subset(dataset, ids)
        print(len(dataset), "\n\n\n")
    elif args.data_set == 'CIFAR100SUBSET':
        dataset = datasets.CIFAR100(args.data_path, train=is_train, transform=transform, download=True)
        nb_classes = 100
        if is_train:
            with open(args.subset_ids, 'r') as file:
                ids = json.load(file)
            dataset = torch.utils.data.Subset(dataset, ids)
        print(len(dataset), "\n\n\n")
    elif args.data_set == 'IMNETSUBSET':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
        if is_train:
            with open(args.subset_ids, 'r') as file:
                ids = json.load(file)
            dataset = torch.utils.data.Subset(dataset, ids)
        print(len(dataset), "\n\n\n")
    elif args.data_set == 'IMNETSUBSET_Generate':
        with open(args.subset_ids, 'r') as file:
            subset_ids = json.load(file)
        # The path is the generated images path
        dataset = ImageNet_Generate_Dataset(args.data_Gen_path, transform=transform, subset_ids = subset_ids)
        nb_classes = 1000
        print(len(dataset), "\n\n\n")

    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * args.input_size)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
