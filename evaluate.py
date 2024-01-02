import argparse
import datetime
import time
import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import json
import os
from scipy.ndimage import zoom
from pathlib import Path
from torch.utils.data import Dataset
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler
from torchvision.transforms import Resize,CenterCrop
from PIL import Image
import random
import numpy as np
import h5py
from medpy import metric
import sys
import models
import cv2
from matplotlib import pyplot as plt

def _crf_with_alpha(cam, alpha, orig_img):
    from imutils import crf_inference
    orig_img = 255*orig_img
    orig_img = np.dstack([orig_img]*3)
    cam = cam.reshape(1,orig_img.shape[0],orig_img.shape[0])
    orig_img = orig_img.astype(np.uint8)
    v = np.array(cam)
    bg_score = np.power(1 - v, alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    return crf_score


def show_cam_on_image(img, mask,pre, save_path,ispre):
    img = np.uint8(img* 255)
    img = np.dstack([img]*3)
    heatmap = cv2.applyColorMap(np.uint8(mask), cv2.COLORMAP_JET)
    add_img = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0,dtype = cv2.CV_32F) 
    if ispre:
        cv2.imwrite(save_path, pre)
    else:
        cv2.imwrite(save_path, add_img)

def calculate_iou(a, b, epsilon=1e-5):
    a = (a > 0).astype(int)
    b = (b > 0).astype(int)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)
    union = np.logical_or(a, b)
    union = np.sum(union)
    iou = intersection / (union + epsilon)
    
    return iou


def mmnormal(x):
    if x.max()-x.min() !=0:
        normal  = (x-x.min())/(x.max()-x.min())
        return normal
    else:
        return x

def calculate_metric(pred, gt):
    dice = metric.binary.dc(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    assd = metric.binary.assd(pred, gt)
    sen = metric.binary.sensitivity(pred, gt)
    spe = metric.binary.specificity(pred, gt)
    iou = calculate_iou(pred,gt)
    return dice, hd95, assd, iou, sen, spe

def getErase(cam, patchatt,cls,n_layers):
        block,batch,h,w = patchatt.shape  # 8x128x196x196
        patch_attn = torch.sum(patchatt[n_layers-8:,:,:,:], dim=0)  # 128x196x196
        patch_cam = torch.reshape(patch_attn,(batch, h, int(h**0.5), int(h**0.5)))  # 128x196x14x14
        weight = mmnormal(torch.reshape(torch.sum(patch_attn[:,:,:],dim=1),(batch,h,1,1)))  # 128x196x1x1
        weighted = torch.sum(patch_cam * weight, dim=1).reshape(batch,h) # 128x14x14  -> 128,196
        cls_cam = mmnormal(cls.reshape(batch,h).cpu().detach().numpy())
        weighted = mmnormal(weighted.cpu().detach().numpy())

        #############  erase model   ############
        weighted_erase = np.ones((batch,h)).astype(np.float32)
        thresh = np.percentile(weighted,95)
        weighted_erase[cls_cam>=thresh] = 0
        weighted_erase.reshape(batch,h,1)

        weighted_erase = torch.from_numpy(weighted_erase.reshape(batch,h,1)).to('cuda')
        weightsum = weighted_erase.sum()

        return weighted, cls_cam, weighted_erase

class BaseDataSet(Dataset):
    def __init__(
        self,
        base_dir=None,
        data_ls = None,):
        self._base_dir = base_dir
        self.ls = data_ls 
        with open(self._base_dir + self.ls, "r") as f1:
            self.sample_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        print("seg dataset  total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(self._base_dir + case, "r")
        image = torch.from_numpy(h5f["image"][:])
        label = torch.from_numpy(h5f["label"][:])
        image = torch.unsqueeze(image, 0)
        label = torch.unsqueeze(label, 0)
        center_resize = CenterCrop(500)
        torch_resize = Resize((448,448))
        image = center_resize(image)
        label = center_resize(label)
        image = torch_resize(image)
        label = torch_resize(label)
        sample = {"image": image, "label": label,"case": case}
        return sample   

model = create_model(
        "PeMformer",
        pretrained=False,
        num_classes=1,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        drop_block_rate=None)

checkpoint = torch.load("", map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
model.to('cuda')


db_train = BaseDataSet(base_dir="./rsna", data_ls="/slices.txt")
trainloader = DataLoader(db_train, batch_size=1, shuffle=False,
                             num_workers=16, pin_memory=True)


th_i = 0
while th_i <= 7:
    dc = 0.
    hd = 0.
    assd = 0.
    sum = 0.
    iou = 0.
    sen = 0.
    spe = 0.
    pre_array = []
    label_array = []
    global_pre_array = []
    global_label_array = []
    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(trainloader):
            img, label, case = sampled_batch['image'], sampled_batch['label'], sampled_batch['case'][0].replace('.h5','').split('/')[-1]
            slice_num = int(case.split('_')[2])
            img = img.to('cuda'); label = label.to('cuda')
            erase_weight = torch.ones(img.shape[0],784,1).to('cuda')

            output, patchcls, cls_attentions, patch_attn, cls,patchtoken = model(img,erase_weight = erase_weight, attention_type="fused")
            weighted, cls_cam, weighted_erase = getErase(cls_attentions, patch_attn,cls,n_layers=th_i)
            output = torch.sigmoid(output).cpu().detach().numpy()
            

            inputsize = 448
            patchnum = int(inputsize/16)
            ori_size = (inputsize,inputsize)
            # ori_size = (192,192)
            map_size = (patchnum,patchnum)
            patch_len = patchnum**2
            patch_att = (patch_len,patchnum,patchnum)

            cls_attentions = F.interpolate(cls_attentions, size=ori_size, mode='bilinear', align_corners=False)[0]
            cls = F.interpolate(cls, size=ori_size, mode='bilinear', align_corners=False)[0]
            cls = cls.squeeze(0).cpu().numpy()

            weighted = torch.from_numpy(weighted.reshape(map_size)).unsqueeze(0).unsqueeze(0)
            weighted = F.interpolate(weighted, size=ori_size, mode='bilinear', align_corners=False).squeeze(0).squeeze(0).cpu().numpy()


            pre = weighted
            thresh = 0.5

            img = img.squeeze(0).squeeze(0).cpu().numpy()
            label = label.squeeze(0).squeeze(0).cpu().numpy()
            cam = pre*255

            ispre= False
            if output<=0.8:
               pre = np.zeros_like(pre)
            #    show_cam_on_image(img,cam,pre,"res/{}.png".format(i_batch),ispre)

            elif pre.sum()<20000:
                pre[pre < thresh] = 0
                pre[pre >= thresh] = 1
                # show_cam_on_image(img,cam,pre*255,"res/{}.png".format(i_batch),ispre)
            elif pre.sum()>20000:
                pre = np.zeros_like(pre)
                # show_cam_on_image(img,cam,pre,"res/{}.png".format(i_batch),ispre)
                continue


            # print(pre.sum())
            if slice_num == 0 and i_batch!=0:
                pre_array = np.array(pre_array)
                label_array = np.array(label_array)
                if pre_array.sum()==0:
                    dice, hd95, asd, iouu, senn, spee = 0,0,0,0,0,0
           # if pre.sum() > 0 and label.sum()>0:
                else:
                    dice, hd95, asd, iouu, senn, spee = calculate_metric(pre_array, label_array)

                dc += dice
                hd += hd95
                assd += asd
                iou += iouu
                sen += senn
                spe += spee
                sum +=1
                pre_array = []
                label_array = []
            pre_array.append(pre.tolist())
            label_array.append(label.tolist())
        print("threshold :{:.5f}, mean dice: {:.5f}, hd95: {:.5f},  assd: {:.5f} iou: {:.5f}, sen: {:.5f}, spe :{:.5f}".format(th_i, dc/sum, hd/sum, assd/sum, iou/sum, sen/sum, spe/sum))
        th_i += 1
