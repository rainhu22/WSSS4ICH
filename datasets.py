import os
from torchvision import transforms
import random
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import PIL.Image
from scipy import ndimage
import math




class RSNAClsDataset(Dataset):
    def __init__(self, img_name_list_path, rsna_root,train=True, transform=None, gen_attn=False):
        img_name_list_path = os.path.join(img_name_list_path, f'{"train" if train or gen_attn else "val"}.csv')
        self.img_name_list = np.array(pd.read_csv(img_name_list_path))
        self.rsna_root = rsna_root
        self.transform = transform
        self.train = train
        self.gen_attn = gen_attn

    def __getitem__(self, idx):
        name = self.img_name_list[idx][0]
        img = PIL.Image.open(os.path.join(self.rsna_root, name + '.png'))
        label = self.img_name_list[idx][1:2].astype("float32")
        label = torch.from_numpy(label)#.unsqueeze(0)

        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.img_name_list)
    


def build_dataset(is_train, args, gen_attn=False):
    transform = build_transform(is_train, args)
    dataset = None
    nb_classes = None
    if args.data_set == 'RSNA':
        dataset = RSNAClsDataset(img_name_list_path=args.img_list, rsna_root=args.data_path,train=is_train, gen_attn=gen_attn, transform=transform)
        nb_classes = 1
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    t = []
    if resize_im and not args.gen_attention_maps:
        size = args.input_size
        # t.append(transforms.CenterCrop(480))
        t.append(transforms.Resize(size, interpolation=3))  # to maintain same ratio w.r.t. 224 images
        t.append(transforms.CenterCrop(args.input_size))
    if is_train:
        t.append(transforms.RandomRotation(degrees=90))
        t.append(transforms.RandomHorizontalFlip(p=0.5))
        t.append(transforms.RandomVerticalFlip(p=0.5))
        t.append(transforms.ColorJitter(brightness=0.3, contrast=0.3))
        

    t.append(transforms.ToTensor())
    t.append(transforms.RandomErasing(p=0.5, scale=(0.01, 0.05), ratio=(0.3, 3.3)))
    return transforms.Compose(t)

