""" Find the indices of CIFAR100 with given imbalance ratio """
import numpy as np
import os
def get_img_num_per_cls(cls_num, imb_type, imb_factor):
    img_max = 50000 / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

def find_indices(labels, imb_type='exp', imb_ratio = 10, cls_num = 100):
    img_num_per_cls = get_img_num_per_cls(cls_num, imb_type, 1. / imb_ratio)

    targets_np = np.array(labels, dtype=np.int64)
    classes = np.unique(targets_np)

    selected_indices = []
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        idx = np.where(targets_np == the_class)[0]
        # np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        # print(the_class, the_img_num, selec_idx[:10])
        selected_indices.extend(selec_idx)

    return selected_indices

feature_path = '../features/CIFAR100_train.npy'
inputs = np.load(feature_path)
features, labels = inputs[:, :-1], inputs[:, -1]
print(features.shape, labels.shape)
print(labels[:10])

for imb_ratio in [10, 50, 100]:
    selected_indices = find_indices(labels, imb_ratio=imb_ratio, cls_num=100)
    print(imb_ratio, len(selected_indices))

    # output_path = '../features/Extra_Dataset/CIFAR100LT/'
    # if not os.path.exists(output_path):
    #     os.makedirs(output_path)
    #
    # output_feature_path = os.path.join(output_path, f'CIFAR100LT_IR{imb_ratio}.npy')
    # np.save(output_feature_path, inputs[selected_indices])
    #
    # output_indices_path = os.path.join(output_path, f'CIFAR100LT_IR{imb_ratio}_indices.npy')
    # np.save(output_indices_path, selected_indices)
