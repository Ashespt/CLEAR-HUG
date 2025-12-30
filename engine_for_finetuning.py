import copy
import math
import sys
from typing import Iterable, Optional
import torch
from timm.utils import ModelEma
import utils.utils as utils
from einops import rearrange
import torch.distributed as dist
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_auc_score,
)

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
    attn_mask = torch.zeros([1, (L+cls_token_num), (L+cls_token_num)])
    attn_mask[:, :cls_token_num, :cls_token_num] = torch.eye(cls_token_num).int().clone()
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

def train_class_batch(
    model, samples, target, criterion, in_chan_matrix, in_time_matrix, key_padding_mask, attn_mask=None
):
    outputs = model(
        samples, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix,key_padding_mask=key_padding_mask,attn_mask=attn_mask
    )
    loss = criterion(outputs, target)
    return loss, outputs


def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return (
        optimizer.loss_scale
        if hasattr(optimizer, "loss_scale")
        else optimizer.cur_scale
    )


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    max_norm: float = 0,
    model_ema: Optional[ModelEma] = None,
    log_writer=None,
    start_steps=None,
    lr_schedule_values=None,
    wd_schedule_values=None,
    num_training_steps_per_epoch=None,
    update_freq=None,
    is_binary=True,
):
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    metric_logger.add_meter(
        "min_lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}")
    )
    header = "Epoch: [{}]".format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    all_outputs = []
    all_targets = []
    selected = [6,7,8,9,10,11]

    for data_iter_step, (samples, targets, in_chan_matrix, in_time_matrix, mask_pad_matrix) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if (
            lr_schedule_values is not None
            or wd_schedule_values is not None
            and data_iter_step % update_freq == 0
        ):
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get(
                        "lr_scale", 1.0
                    )
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]

        samples = samples.float().to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask_pad_matrix = mask_pad_matrix.to(device, non_blocking=True).bool()
        bool_masked_pos, attn_mask = random_masking_attn_mask(samples, mask_ratio=0, cls_token_num=12)

        if is_binary:
            targets = targets.float()
        # import pdb;pdb.set_trace()
        # samples=samples.reshape((samples.shape[0],12,15,96))
        # samples[:,selected,:,:] = 0
        # samples=samples.reshape((samples.shape[0],-1,96))
        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, in_chan_matrix, in_time_matrix, key_padding_mask=mask_pad_matrix,attn_mask=attn_mask
            )
        else:
            with torch.cuda.amp.autocast(enabled=False): #enabled=False
                loss, output = train_class_batch(
                    model, samples, targets, criterion, in_chan_matrix, in_time_matrix, key_padding_mask=mask_pad_matrix,attn_mask=attn_mask
                )

        # utils.save_tensor_to_file(
        #     output, "log/temp/outputs_train.txt", append=True
        # )

        if is_binary:
            output = torch.sigmoid(output) # Compute Sigmoid Results for AUC-ROC Calculation
        else:
            output = torch.softmax(output)

        loss_value = loss.item()

        # if not math.isfinite(loss_value):
        #     print("Loss is {}, stopping training".format(loss_value))
        #     sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = (
                hasattr(optimizer, "is_second_order") and optimizer.is_second_order
            )
            loss /= update_freq
            grad_norm = loss_scaler(
                loss,
                optimizer,
                clip_grad=max_norm,
                parameters=model.parameters(),
                create_graph=is_second_order,
                update_grad=(data_iter_step + 1) % update_freq == 0,
            )
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        all_outputs.append(output.detach())
        all_targets.append(targets.detach())

        torch.cuda.synchronize()

        if is_binary:
            class_acc = utils.analyze_ecg_classification(
                output.detach().cpu().numpy(),
                targets.detach().cpu().numpy(),
                is_binary,
            )["accuracy"]
        else:
            class_acc = (output.max(-1)[-1] == targets.squeeze()).float().mean()

        metric_logger.update(loss=loss_value)
        metric_logger.update(class_acc=class_acc)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.0
        max_lr = 0.0
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
            log_writer.update(loss=loss_value, head="train")
            log_writer.update(class_acc=class_acc, head="train")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # all_outputs = torch.cat(all_outputs, dim=0)
    # all_targets = torch.cat(all_targets, dim=0)

    # gather_all_outputs = utils.gather_tensor(all_outputs)
    # gather_all_targets = utils.gather_tensor(all_targets)
    # roc_auc = utils.compute_roc_auc(
    #     gather_all_outputs.cpu().numpy(),
    #     gather_all_targets.cpu().numpy(),
    # )
    # metric_logger.update(roc_auc=roc_auc)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    data_loader,
    model,
    device,
    header="Test:",
    metrics=["acc"],
    is_binary=True,
    dataset_dir="",
):
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    selected = [6,7,8,9,10,11]
    with torch.no_grad():
        for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
            ECG = batch[0]
            # ECG=ECG.reshape((ECG.shape[0],12,15,96))
            # ECG[:,selected,:,:] = 0
            # ECG=ECG.reshape((ECG.shape[0],-1,96))
            target = batch[1]
            in_chan_matrix = batch[2]
            in_time_matrix = batch[3]
            if len(batch) == 5:
                mask_pad_matrix = batch[4].to(device, non_blocking=True).bool()
            else:
                mask_pad_matrix = None
            bool_masked_pos, attn_mask = random_masking_attn_mask(ECG, mask_ratio=0,cls_token_num=12)
            ECG = ECG.float().to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if is_binary:
                target = target.float()

            # compute output
            with torch.cuda.amp.autocast(enabled=False): #enabled=False
                output = model(
                    ECG, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix,key_padding_mask=mask_pad_matrix,attn_mask=attn_mask
                )
                loss = criterion(output, target)
    
            if is_binary:
                output = torch.sigmoid(output).cpu()
            else:
                output = torch.softmax(output).cpu()

            target = target.cpu()

            pred.append(output)
            true.append(target)
            results = utils.analyze_ecg_classification(
                output.numpy(), target.numpy(), is_binary, threshold=0.5
            )

            batch_size = ECG.shape[0]
            metric_logger.update(loss=loss.item())
            for key, value in results.items():
                metric_logger.meters[key].update(value, n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # print when running
    all_outputs = torch.cat(pred, dim=0).to(device)
    all_targets = torch.cat(true, dim=0).to(device)

    if torch.cuda.device_count() > 1:
        gather_all_outputs = utils.gather_tensor(all_outputs)
        gather_all_targets = utils.gather_tensor(all_targets)
        print(f"gather_all_outputs shape: {gather_all_outputs.shape}")
        print(f"gather_all_targets shape: {gather_all_targets.shape}")
    else:
        gather_all_outputs = copy.deepcopy(all_outputs)
        gather_all_targets = copy.deepcopy(all_targets)

    if header == "Test:":
        parts = dataset_dir.strip("/").split("/")
        if len(parts) >= 2:
            dataset_name = "_".join(parts[-2:])
        else:
            dataset_name = "_".join(parts)

        output_file_path = f"results/pred/{dataset_name}_outputs.csv"
        target_file_path = f"results/pred/{dataset_name}_targets.csv"

        utils.save_tensor_to_csv(gather_all_outputs, output_file_path)
        utils.save_tensor_to_csv(gather_all_targets, target_file_path)
    
    roc_auc = utils.compute_roc_auc(
        gather_all_outputs.cpu().numpy(),
        gather_all_targets.cpu().numpy(),
    )
    metric_logger.update(roc_auc=roc_auc)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("* loss {losses.global_avg:.3f}".format(losses=metric_logger.loss))

    ret = utils.analyze_ecg_classification(
        all_outputs.cpu().numpy(), all_targets.cpu().numpy(), is_binary, 0.5
    )
    ret["loss"] = metric_logger.loss.global_avg
    ret["roc_auc"] = roc_auc
    return ret


@torch.no_grad()
def to_class(
    data_loader,
    model,
    device,
    header="Test:",
    metrics=["acc"],
    is_binary=True,
    dataset_dir="",
):
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")

    # switch to evaluation mode
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
            ECG = batch[0]
            
            target = batch[1]
            in_chan_matrix = batch[2]
            in_time_matrix = batch[3]
            if len(batch) == 5:
                mask_pad_matrix = batch[4].to(device, non_blocking=True).bool()
            else:
                mask_pad_matrix = None
            bool_masked_pos, attn_mask = random_masking_attn_mask(ECG, mask_ratio=0,cls_token_num=12)
            ECG = ECG.float().to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            if is_binary:
                target = target.float()

            # compute output
            with torch.cuda.amp.autocast(enabled=False): #enabled=False
                output = model(
                    ECG, in_chan_matrix=in_chan_matrix, in_time_matrix=in_time_matrix,key_padding_mask=mask_pad_matrix,attn_mask=attn_mask
                )
                loss = criterion(output, target)
    
            if is_binary:
                output = torch.sigmoid(output).cpu()
            else:
                output = torch.softmax(output).cpu()

            target = target.cpu()

            pred.append(output)
            true.append(target)
            results = utils.analyze_ecg_classification(
                output.numpy(), target.numpy(), is_binary, threshold=0.5
            )

            batch_size = ECG.shape[0]
            metric_logger.update(loss=loss.item())
            for key, value in results.items():
                metric_logger.meters[key].update(value, n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # print when running
    all_outputs = torch.cat(pred, dim=0).to(device)
    all_targets = torch.cat(true, dim=0).to(device)

    if torch.cuda.device_count() > 1:
        gather_all_outputs = utils.gather_tensor(all_outputs)
        gather_all_targets = utils.gather_tensor(all_targets)
        print(f"gather_all_outputs shape: {gather_all_outputs.shape}")
        print(f"gather_all_targets shape: {gather_all_targets.shape}")
    else:
        gather_all_outputs = copy.deepcopy(all_outputs)
        gather_all_targets = copy.deepcopy(all_targets)

    if header == "Test:":
        parts = dataset_dir.strip("/").split("/")
        if len(parts) >= 2:
            dataset_name = "_".join(parts[-2:])
        else:
            dataset_name = "_".join(parts)

        output_file_path = f"results/pred/{dataset_name}_outputs.csv"
        target_file_path = f"results/pred/{dataset_name}_targets.csv"

        utils.save_tensor_to_csv(gather_all_outputs, output_file_path)
        utils.save_tensor_to_csv(gather_all_targets, target_file_path)

    gather_all_outputs = gather_all_outputs.cpu().numpy()
    gather_all_targets = gather_all_targets.cpu().numpy()
    roc_auc,best_threshold_per_label = utils.compute_roc_auc_per_label(
        gather_all_outputs,
        gather_all_targets,
    )
    
    gather_all_outputs_thres = np.zeros_like(gather_all_outputs)
    for i, thres in enumerate(best_threshold_per_label):
        gather_all_outputs_thres[:,i] = (gather_all_outputs[:,i] > thres).astype(int)
    
    total_accuracy = accuracy_score(gather_all_targets, gather_all_outputs_thres)

