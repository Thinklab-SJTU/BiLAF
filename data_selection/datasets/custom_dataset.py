import torch
import os
import json
from PIL import Image, ImageFile
from collections import Counter


class MPIIReturnIndexCountDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(MPIIReturnIndexCountDataset, self).__init__()
        self.image_root = os.path.join(root, "images")
        ann_file = os.path.join(root, "annotations", "mpii_train.json")

        with open(ann_file, "r") as file:
            anns = json.load(file)

        filenames = []
        for ann in anns:
            filenames.append(ann["image"])
        self.count = Counter(filenames)
        print("max count:", max(list(self.count.values())))
        self.filenames = list(set(filenames))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.image_root, filename)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, filename, self.count[filename]


class NYUReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(NYUReturnIndexDataset, self).__init__()
        self.image_root = os.path.join(root, "NYU_Depth_V2/sync")
        ann_file = os.path.join(root, "nyudepthv2_train_files_with_gt_dense.txt")

        with open(ann_file, "r") as file:
            self.lines = file.readlines()

        self.transform = transform

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx]
        image_path, _, _ = line.split(" ")
        path = os.path.join(self.image_root, image_path)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, line


class CrowdPoseReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(CrowdPoseReturnIndexDataset, self).__init__()
        self.image_root = os.path.join(root, "images")
        ann_file = os.path.join(root, "annotations", "mmpose_crowdpose_trainval.json")

        with open(ann_file, "r") as file:
            anns = json.load(file)
        self.anns = anns["images"]

        self.transform = transform

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        path = os.path.join(self.image_root, ann["file_name"])
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, ann["id"]


class TinyImageNetReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(TinyImageNetReturnIndexDataset, self).__init__()
        self.root = root
        id_file = os.path.join(root, "train", "train.txt")
        with open(id_file, "r") as file:
            ids_ = file.readlines()
        ids = []
        for i in range(len(ids_)):
            ids.append(ids_[i].strip())

        self.ids = list(filter(lambda id: len(id)>0, ids))
        print("\nData Size", len(ids), "\n")

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        prefix, filename, gt = id.split(" ")
        path = os.path.join(self.root, "train", prefix, filename)
        with open(path, 'rb') as f:
            try:
                img = Image.open(f)
                img = img.convert('RGB')
            except:
                print(id, "\n\n\n\n\n\n")

        if self.transform is not None:
            img = self.transform(img)
        return img, id


class OxfordPetReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(OxfordPetReturnIndexDataset, self).__init__()
        self.root = root
        self.image_root = os.path.join(root, "images")
        id_file = os.path.join(root, "annotations", "trainval.txt")
        with open(id_file, "r") as file:
            ids_ = file.readlines()
        ids = []
        for i in range(len(ids_)):
            ids.append(ids_[i].strip())

        self.ids = list(filter(lambda id: len(id)>0, ids))
        print("\nData Size", len(ids), "\n")

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        filename = id.split(" ")[0] + ".jpg"
        path = os.path.join(self.image_root, filename)
        with open(path, 'rb') as f:
            try:
                img = Image.open(f)
                img = img.convert('RGB')
            except:
                print(id, "\n\n\n\n\n\n")

        if self.transform is not None:
            img = self.transform(img)
        return img, id


class MPIIReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(MPIIReturnIndexDataset, self).__init__()
        self.image_root = os.path.join(root, "images")
        ann_file = os.path.join(root, "annotations", "mpii_train.json")

        with open(ann_file, "r") as file:
            anns = json.load(file)

        filenames = []
        for ann in anns:
            filenames.append(ann["image"])
        self.filenames = list(set(filenames))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.image_root, filename)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, filename


class COCOReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, panoptic, transform):
        super(COCOReturnIndexDataset, self).__init__()
        self.image_root = os.path.join(root, "train2017")
        if panoptic:
            ann_file = os.path.join(root, "annotations", "panoptic_train2017.json")
        else:
            ann_file = os.path.join(root, "annotations", "instances_train2017.json")

        with open(ann_file, "r") as file:
            anns = json.load(file)
        self.anns = anns["images"]

        self.transform = transform

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        path = os.path.join(self.image_root, ann["file_name"])
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, ann["id"]


class CaltechReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(CaltechReturnIndexDataset, self).__init__()
        self.root = root
        self.image_root = os.path.join(root, "256_ObjectCategories")
        id_file = os.path.join(root, "trainval.txt")
        with open(id_file, "r") as file:
            ids_ = file.readlines()
        ids = []
        for i in range(len(ids_)):
            ids.append(ids_[i].strip())

        self.ids = list(filter(lambda id: len(id)>0, ids))
        print("\nData Size", len(ids), "\n")

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        path = os.path.join(self.image_root, id)
        with open(path, 'rb') as f:
            try:
                img = Image.open(f)
                img = img.convert('RGB')
            except:
                print(id, "\n\n\n\n\n\n")

        if self.transform is not None:
            img = self.transform(img)
        return img, id


class CitySegReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(CitySegReturnIndexDataset, self).__init__()
        self.root = root
        self.image_root = os.path.join(root, "leftImg8bitJPG/train")
        id_file = os.path.join(root, "train.txt")
        with open(id_file, "r") as file:
            ids_ = file.readlines()
        ids = []
        for i in range(len(ids_)):
            ids.append(ids_[i].strip())

        self.ids = list(filter(lambda id: len(id)>0, ids))
        print("\nData Size", len(ids), "\n")

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        path = os.path.join(self.image_root, "%s_leftImg8bit.jpg"%id)
        with open(path, 'rb') as f:
            try:
                img = Image.open(f)
                img = img.convert('RGB')
            except:
                print(id, "\n\n\n\n\n\n")

        if self.transform is not None:
            img = self.transform(img)
        return img, id


class CityReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(CityReturnIndexDataset, self).__init__()
        self.image_root = os.path.join(root, "leftImg8bit/train")
        ann_file = os.path.join(root, "annotations", "instancesonly_filtered_gtFine_train.json")

        with open(ann_file, "r") as file:
            anns = json.load(file)
        self.anns = anns["images"]

        self.transform = transform

    def __len__(self):
        return len(self.anns)

    def __getitem__(self, idx):
        ann = self.anns[idx]
        path = os.path.join(self.image_root, ann["file_name"])
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, ann["id"]


class ReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, year, transform):
        super(ReturnIndexDataset, self).__init__()
        id_file = os.path.join(root, "VOCdevkit/VOC%s/ImageSets/Main"%year, "trainval.txt")
        with open(id_file, "r") as file:
            ids = file.readlines()
        for i in range(len(ids)):
            ids[i] = ids[i].strip()

        self.ids = list(filter(lambda id: len(id)>0, ids))
        print("\nData Size", len(ids), "\n")

        self.root = root
        self.year = year
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        path = os.path.join(self.root, "VOCdevkit/VOC%s"%self.year, "JPEGImages", "%s.jpg"%id)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, id



class KITTILapReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(KITTILapReturnIndexDataset, self).__init__()
        id_file = os.path.join(root, "eigen_train_files_with_gt_dense.txt")
        self.root = os.path.join(root, "KITTI")

        with open(id_file, "r") as file:
            ids_ = file.readlines()
        self.ids = []
        for id in ids_:
            line = id.split(" ")
            if line[1] == "None":
                print(line)
                continue
            self.ids.append(id.strip())

        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        line = self.ids[idx]
        line_ =line.split()
        path = os.path.join(self.root, line_[0])

        with open(path, 'rb') as f:
            try:
                img = Image.open(f)
                img = img.convert('RGB')
            except:
                print(id, "\n\n\n\n\n\n")

        if self.transform is not None:
            img = self.transform(img)
        return img, line


class ADEReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform):
        super(ADEReturnIndexDataset, self).__init__()
        self.filenames = os.listdir(root)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        name = filename[:-4]

        path = os.path.join(self.root, filename)

        with open(path, 'rb') as f:
            try:
                img = Image.open(f)
                img = img.convert('RGB')
            except:
                print(id, "\n\n\n\n\n\n")

        if self.transform is not None:
            img = self.transform(img)
        return img, name

class VOCReturnIndexDataset(torch.utils.data.Dataset):
    def __init__(self, root, year, transform):
        super(VOCReturnIndexDataset, self).__init__()
        id_file = os.path.join(root, "VOCdevkit/VOC%s/ImageSets/Main"%year, "trainval.txt")
        with open(id_file, "r") as file:
            ids = file.readlines()
        for i in range(len(ids)):
            ids[i] = ids[i].strip()

        self.ids = list(filter(lambda id: len(id)>0, ids))
        print("\nData Size", len(ids), "\n")

        self.root = root
        self.year = year
        self.transform = transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        path = os.path.join(self.root, "VOCdevkit/VOC%s"%self.year, "JPEGImages", "%s.jpg"%id)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        return img, id