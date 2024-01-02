import math
import sys
from typing import Iterable
import numpy as np
from scipy.ndimage import gaussian_filter1d
import torch
import torch.nn.functional as F
import utils
import torch.nn as nn
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score, f1_score, hamming_loss, roc_auc_score, precision_score, recall_score,auc, accuracy_score
from loss import InfoNCE
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
from pathlib import Path
import warnings
import pandas as pd
import json
warnings.filterwarnings("ignore")

    
def mmnormal(x):
    normal  = (x-x.min())/(x.max()-x.min())
    return normal

def getErase(cam, patchatt,cls):
        block,batch,h,w = patchatt.shape  # 8x128x196x196
        patch_attn = torch.sum(patchatt[:,:,:,:], dim=0)  # 128x196x196
        patch_cam = torch.reshape(patch_attn,(batch, h, int(h**0.5), int(h**0.5)))  # 128x196x14x14
        weight = mmnormal(torch.reshape(torch.sum(patch_attn[:,:,:],dim=1),(batch,h,1,1)))  # 128x196x1x1
        weighted = torch.sum(patch_cam * weight, dim=1).reshape(batch,h) # 128x14x14  -> 128,196
        masked_weight = mmnormal((cam.reshape(batch,h)*weighted).cpu().detach().numpy())
        weighted = mmnormal(weighted.cpu().detach().numpy())

        #############  erase model   ############
        weighted_erase = F.interpolate(cls*cam, size=(16*int(h ** 0.5),16*int(h ** 0.5)), mode='bilinear', align_corners=False)
        thresh = np.percentile(masked_weight.detach().cpu().numpy(),95)
        weighted_erase[weighted_erase<=thresh] = 1
        weighted_erase[weighted_erase<1] = 0
        weighted_erase = weighted_erase.to('cuda')

        return masked_weight, weighted_erase



criterion = torch.nn.BCEWithLogitsLoss()
nce = InfoNCE()

def train_one_epoch(output_dir, model: torch.nn.Module, data_loader: Iterable,
                    optimizer: torch.optim.Optimizer, device: torch.device,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    batch_num = 0


    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)  # 128x1
        erase_weight = torch.ones_like(samples).to('cuda')

        patch_outputs = None
        batch = samples.shape[0]
        batch_idx = np.nonzero(targets.cpu().detach().numpy().reshape((batch)))[0]
        with torch.cuda.amp.autocast():
            outputs = model('transcam',samples,erase_weight)
            if not isinstance(outputs, torch.Tensor):
                # outputs, patch_outputs, cam, patchatt,cls, patch_tokens = outputs  # PEMformer
                output, patch_outputs, conv_cam, cls, atten, patch_tokens  = outputs  # Conformer

            ploss = criterion(patch_outputs, targets)
            metric_logger.update(pat_loss=ploss.item())
            cls_loss = criterion(output, targets)
            metric_logger.update(cls_loss=cls_loss.item())
            loss = cls_loss+ploss

            ################################### Erase###############################
            if epoch>=0:
                weighted, weighted_erase = getErase(conv_cam, atten,cls)
                weightedsum = weighted_erase.sum()  
                outputs2, patch_outputs2, conv_cam2, cls2, atten2, patch_tokens2 = model('transcam',samples,erase = weighted_erase )
                weighted2, weighted_erase2 = getErase(conv_cam2, atten2,cls2)  
                out1, out2 = torch.sigmoid(output).detach().cpu().numpy(),torch.sigmoid(outputs2).detach().cpu().numpy()
                out1 = np.where(out1>=0.5,1,0); out2 = np.where(out2>=0.5,1,0) 
                out1_tar = out1*targets.detach().cpu().numpy()  # 与label相乘后的
                true_pos = out1 - out2  
                target_tp = torch.from_numpy(np.where(true_pos==1,0,targets.detach().cpu().numpy())).to('cuda')
                target_mask = np.nonzero(np.where(true_pos==1,1,0))[0]
                mask = torch.ones(targets.shape[0], dtype=torch.bool)
                mask[target_mask] = False
                masked_targets = targets.clone()[mask]
                outputs2 = outputs2[mask]   
                cls_loss2 = criterion(outputs2, masked_targets)
                metric_logger.update(cls_loss2=cls_loss2.item())
                loss = loss + cls_loss2
                loss = cls_loss2


        ###################  NCE Loss  #######################
            if epoch>=0:
                batch_idx = np.nonzero(target_tp.cpu().detach().numpy().reshape((batch)))[0]
                # thresh = np.percentile(weighted.cpu().detach().numpy(),99)

                weight_1 = weighted.copy()
                weight_2 = weighted2.copy()
                weight_neg = (weighted.copy()+weighted2.copy())/2
                # patch_tokens = patch_tokens.detach()

                weight_neg[weight_neg<0.05] = 0
                weight_neg[weight_neg>0.3] = 0
                weight_neg = weight_neg.tolist()
                weight_1[weight_1<0.5] = 0  # 0.5
                weight_2[weight_2<0.5] = 0  # 0.5

                query = torch.zeros(1,192).to('cuda')
                possitive = torch.zeros(1,192).to('cuda')
                negtive = torch.zeros(1,192).to('cuda')

                if len(batch_idx)>=1:
                    for i in batch_idx:
                        idx_1 = torch.topk(torch.from_numpy(weight_1[i]),3)[1][:2]
                        idx2 = torch.topk(torch.from_numpy(weight_2[i]),3)[1][:2]
                        if len(idx_1) == 0 or len(idx2) == 0:
                            continue
                        for j in idx_1:
                            query = torch.cat((query,patch_tokens[i][j].clone().reshape(1,192)))
                        for k in idx2:
                            possitive = torch.cat((possitive,patch_tokens2[i][k].clone().reshape(1,192)))
                        idx_neg = np.nonzero(weight_neg[i])[0]
                        if len(idx_neg) >=10:
                            idx_neg = np.random.choice(idx_neg,size=10,replace = False)
                        else:
                            pass
                        for n in idx_neg:
                            negtive = torch.cat((negtive,patch_tokens[i][n].clone().reshape(1,192)))

                query =  query[1:,:]
                negtive =  negtive[1:,:]
                possitive = possitive[1:,:]
                q_len = query.shape[0]


                if len(batch_idx) and q_len >1:
                    nce_loss = nce(query,possitive,negtive)
                    metric_logger.update(nce_loss=0.2*nce_loss.item())
                    loss = loss + 0.2*nce_loss


        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f'train_{k}': format(v,".4f") for k, v in train_stats.items()}}
        
        if batch_num % 50 == 0:
            with (output_dir / "train_metric.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
        batch_num += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return train_stats


@torch.no_grad()  # 等同于with torch.no_grad()
def evaluate(output_dir, data_loader, model, device):
    mAP = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    batch_num = 0
    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_size = images.shape[0]
        erase_weight = torch.ones_like(images).to('cuda')

        with torch.cuda.amp.autocast():
            # output = model(images,erase_weight = erase_weight)
            output = model("transcam",images, erase_weight)
            if not isinstance(output, torch.Tensor):
                # output, patch_output,cam, patchatt, cls, patch_tokens = output
                outputs, patch_outputs, conv_cam, cls, atten, patch_tokens = output
            ploss = criterion(patch_outputs, target)
            metric_logger.update(pat_loss=ploss.item())
            cls_loss = criterion(outputs, target)
            metric_logger.update(cls_loss=cls_loss.item())
            loss = cls_loss+ploss
            loss = criterion(outputs, target)

            output = torch.sigmoid(outputs)

            p,r,f1,acc = compute_mAP(target, output)
            metric_logger.meters['precision'].update(p, n=batch_size)
            metric_logger.meters['recall'].update(r, n=batch_size)
            metric_logger.meters['f1_score'].update(f1, n=batch_size)
            metric_logger.meters['acc'].update(acc, n=batch_size)


        metric_logger.update(loss=loss.item())

    test_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    log_stats = {**{f'test_{k}': format(v,".4f") for k, v in test_stats.items()}}
        # if batch_num %10 == 0:
    with (output_dir / "test_metric.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")
        # batch_num += 1


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('* acc {acc.global_avg:.3f} loss {losses.global_avg:.3f}'
              .format(acc=metric_logger.acc, losses=metric_logger.loss))


    return test_stats


def compute_mAP(labels, outputs):
    y_true = labels.cpu().numpy()
    y_pred = outputs.cpu().numpy()
    y_pred[y_pred>0.5]=1
    y_pred[y_pred<=0.5]=0
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision,recall,f1,acc


def _crf_with_alpha(cam_dict, alpha, orig_img):
    from imutils import crf_inference
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    crf_score = crf_inference(orig_img, bgcam_score, labels=bgcam_score.shape[0])

    n_crf_al = dict()

    n_crf_al[0] = crf_score[0]
    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = crf_score[i + 1]

    return n_crf_al
