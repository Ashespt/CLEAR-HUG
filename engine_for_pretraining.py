import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
from contextlib import nullcontext
import utils.utils as utils
import random
from torch_ecg.cfg import CFG
from torch_ecg.augmenters import AugmenterManager
import numpy as np
config = CFG(
    random=False,
    fs=500,
    baseline_wander={},
    random_flip={},
    stretch_compress={},
    # cut_mix={},
    # random_masking={},
    # random_renormalize={},
    # label_smooth={},
    # mixup={},
)
time_warp = AugmenterManager.from_config(config)

def random_masking(x, mask_ratio):
    N, L, D = x.shape
    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(N, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask.to(torch.bool)

def data_aug(data):
    mean = 0.0
    std = 0.5
    samples = torch.randn(data.shape) * std + mean
    data+=samples
    data += np.random.normal(0, 0.01, size=data.shape)
    if np.random.uniform(0,1) >= 0.5:
        scale = np.random.uniform(0.8, 1.2)
        data = data * scale
        # data = -data
    return data

def random_masking_attn_mask(x, mask_ratio,cls_token_num=1):
    N, L, D = x.shape

    num_tokens_plead = L//cls_token_num

    assert num_tokens_plead * cls_token_num == L

    len_keep = int(L * (1 - mask_ratio))

    noise = torch.rand(1, L, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    mask = torch.ones([1, L], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    attn_mask = torch.zeros([1, (L+cls_token_num), (L+cls_token_num)]) #初始化mask
    attn_mask[:, :cls_token_num, :cls_token_num] = torch.eye(cls_token_num).int().clone() # cls token和cls token关系
    cls_mask = torch.zeros([1, cls_token_num, L])
    cls_mask_r = torch.zeros([1, L, cls_token_num])
    for i in range(cls_token_num):
        cls_mask[:, i, num_tokens_plead * i:num_tokens_plead * (i + 1)] = 1
        cls_mask_r[:, num_tokens_plead * i:num_tokens_plead * (i + 1), i] = 1
    attn_mask[:, :cls_token_num, cls_token_num:] = cls_mask
    attn_mask[:, cls_token_num:, :cls_token_num] = cls_mask_r
    for n in range(1):
        mask_idx_copy = torch.where(mask[n] == 1)[0]
        mask_idx = list(mask_idx_copy)
        for l_src in range(L):
            mask_line = torch.zeros(L)
            lead_id = l_src // num_tokens_plead
            if l_src in mask_idx:
                mask_line[l_src % num_tokens_plead::num_tokens_plead] = 1
            else:
                mask_line[:] = 1
                mask_line[mask_idx_copy] = 0

            attn_mask[n, l_src + cls_token_num, cls_token_num:] = mask_line
    
    mask = mask.repeat(N, 1)
    attn_mask = attn_mask.repeat(N, 1, 1)

    return mask.to(torch.bool), ~attn_mask.to(torch.bool)


# def data_aug(data):
#     # data += np.random.normal(0, 0.01, size=data.shape)
#     # if np.random.uniform(0,1) >= 0.5:
#     #     scale = np.random.uniform(0.8, 1.2)
#     #     data = data * scale
#     import pdb;pdb.set_trace()
#     data,_,_ = time_warp(data)
#     # data = -data
#     return data

def train_one_epoch(model: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss().cuda()

    step_loader = 0
    for data_loader in data_loader_list:
        if len(data_loader) == 0:
            continue
        for step, (batch) in enumerate(
                metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            # samples [bs,seq_len,embed_dim]
            samples = batch[0]
            samples = samples.float().to(device, non_blocking=True)
            
            in_chan_matrix = batch[1].to(device, non_blocking=True)
            in_time_matrix = batch[2].to(device, non_blocking=True)
            
            if len(batch) == 4:
                in_pad_mask = batch[3].to(device, non_blocking=True).bool()
            else:
                in_pad_mask = None
            # bool_masked_pos, attn_mask = random_masking_attn_mask(samples, mask_ratio=args.mask_ratio,cls_token_num=args.cls_token_num)
            # bool_masked_pos = bool_masked_pos.to(device, non_blocking=True)
            # attn_mask = attn_mask.to(device, non_blocking=True)
            bool_masked_pos = None
            attn_mask = None
            my_context = model.no_sync if args.distributed and (
                    step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            with my_context():
                with torch.cuda.amp.autocast():  # enabled=False
                    
                    loss,_ = model(samples, mask_bool_matrix=bool_masked_pos, in_chan_matrix=in_chan_matrix,
                                in_time_matrix=in_time_matrix, key_padding_mask=in_pad_mask, return_qrs_tokens=False, \
                                return_all_tokens=False,attn_mask=attn_mask,criterion=None)
                
            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= args.gradient_accumulation_steps
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            
            metric_logger.update(loss=loss.item())

            if log_writer is not None:
                log_writer.update(loss=loss.item(), head="loss")

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

