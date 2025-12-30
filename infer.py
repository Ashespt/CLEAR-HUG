import torch
import torch.nn as nn
import argparse
import datetime
from pyexpat import model
import random
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
from pathlib import Path
import struct
from collections import OrderedDict
from timm.data.mixup import Mixup
from timm.models import create_model
from timm.utils import ModelEma
from utils.optim_factory import (
    create_optimizer,
    get_parameter_groups,
    LayerDecayValueAssigner,
)
from utils.utils import (
    NativeScalerWithGradNormCount as NativeScaler,
    freeze_except_prefix,
    freeze_except_prefixes,
    freeze_specific_layers,
    get_trainable_layers,
    save_results_as_csv,
    set_seed,
)
import utils.utils as utils
from scipy import interpolate
import modeling_finetune as modeling_finetune

from wfdb import processing
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse
import random


class QRSTokenizer(nn.Module):

    def __init__(self, fs, max_len, token_len, lead_len, used_chans=None):
        super(QRSTokenizer, self).__init__()
        self.fs = fs
        self.max_len = max_len
        self.token_len = token_len
        self.lead_len = lead_len
        self.used_chans = used_chans

    def qrs_detection(self, ecg_signal):
        flag = True
        channels, _ = ecg_signal.shape
        all_qrs_inds = []
        lead_signal = ecg_signal[1, :]
        qrs_inds = processing.xqrs_detect(
                sig=lead_signal, fs=self.fs, verbose=False
            )
        if len(qrs_inds) == 0:
            for i in range(12):
                lead_signal = ecg_signal[i, :]
                qrs_inds = processing.xqrs_detect(
                        sig=lead_signal, fs=self.fs, verbose=False
                    )
                if len(qrs_inds) > 0:
                    break
        if len(qrs_inds) == 0:
            # qrs_inds = np.sort(random.sample(range(len(ecg_signal[0])), self.lead_len))
            qrs_inds = np.linspace(0, len(ecg_signal[0]) - 1, self.lead_len, dtype=int)
            flag = False
            
        all_qrs_inds = [qrs_inds]*12

        return all_qrs_inds, flag

    def extract_qrs_segments_old(self, ecg_signal, qrs_inds):
        channels, _ = ecg_signal.shape
        channel_qrs_segments = []
        for channel_index in range(channels):
            qrs_segments = []
            len_segment = max(len(qrs_inds[channel_index]),self.lead_len)
            for i in range(len_segment):
                if i == 0:
                    center = qrs_inds[channel_index][i]
                    start = max(center - self.token_len // 2, 0)
                    if (i + 1) < len(qrs_inds[channel_index]):
                        end = end = (
                            qrs_inds[channel_index][i] + qrs_inds[channel_index][i + 1]
                        ) // 2
                    else:
                        end = min(
                            start + self.token_len, len(ecg_signal[channel_index])
                        )
                elif i == len(qrs_inds[channel_index]) - 1:
                    center = qrs_inds[channel_index][i]
                    end = min(
                        center + self.token_len // 2, len(ecg_signal[channel_index])
                    )
                    start = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i - 1]
                    ) // 2
                elif i > len(qrs_inds[channel_index]) - 1:
                    center = -1
                    start = 1
                    end = 1
                else:
                    center = qrs_inds[channel_index][i]
                    start = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i - 1]
                    ) // 2
                    end = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i + 1]
                    ) // 2
                
                if center < 0:
                    segment = np.zeros(self.token_len)
                    qrs_segments.append(segment)
                    continue

                start = max(start, 0)
                end = min(end, len(ecg_signal[channel_index]))
                actual_len = end - start

                if actual_len > self.token_len:
                    center = qrs_inds[channel_index][i]
                    start = max(center - self.token_len // 2, 0)
                    end = min(start + self.token_len, len(ecg_signal[channel_index]))

                segment = np.zeros(self.token_len)
                segment_start = max(self.token_len // 2 - (center - start), 0)
                segment_end = segment_start + (end - start)

                if segment_end > self.token_len:
                    end -= segment_end - self.token_len
                    segment_end = self.token_len

                # print(f"Segment start: {segment_start}, Segment end: {segment_end}")
                segment[segment_start:segment_end] = ecg_signal[channel_index][
                    start:end
                ]
                qrs_segments.append(segment)
            
            channel_qrs_segments.append(qrs_segments)
        return channel_qrs_segments

    def extract_qrs_segments(self, ecg_signal, qrs_inds):
        channels, _ = ecg_signal.shape
        channel_qrs_segments = []
        channel_pad_masks = []
        for channel_index in range(channels):
            qrs_segments = []
            pad_masks = []
            len_segment = max(len(qrs_inds[channel_index]),self.lead_len)
            for i in range(len_segment):
                if i == 0:
                    center = qrs_inds[channel_index][i]
                    start = max(center - self.token_len // 2, 0)
                    if (i + 1) < len(qrs_inds[channel_index]):
                        end = end = (
                            qrs_inds[channel_index][i] + qrs_inds[channel_index][i + 1]
                        ) // 2
                    else:
                        end = min(
                            start + self.token_len, len(ecg_signal[channel_index])
                        )
                    pad_masks.append(1)
                elif i == len(qrs_inds[channel_index]) - 1:
                    center = qrs_inds[channel_index][i]
                    end = min(
                        center + self.token_len // 2, len(ecg_signal[channel_index])
                    )
                    start = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i - 1]
                    ) // 2
                    pad_masks.append(1)
                elif i > len(qrs_inds[channel_index]) - 1:
                    center = -1
                    start = 1
                    end = 1
                    pad_masks.append(0)
                else:
                    center = qrs_inds[channel_index][i]
                    start = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i - 1]
                    ) // 2
                    end = (
                        qrs_inds[channel_index][i] + qrs_inds[channel_index][i + 1]
                    ) // 2
                    pad_masks.append(1)
                
                if center < 0:
                    segment = np.zeros(self.token_len)
                    qrs_segments.append(segment)
                    continue

                start = max(start, 0)
                end = min(end, len(ecg_signal[channel_index]))
                actual_len = end - start

                if actual_len > self.token_len:
                    center = qrs_inds[channel_index][i]
                    start = max(center - self.token_len // 2, 0)
                    end = min(start + self.token_len, len(ecg_signal[channel_index]))

                segment = np.zeros(self.token_len)
                segment_start = max(self.token_len // 2 - (center - start), 0)
                segment_end = segment_start + (end - start)

                if segment_end > self.token_len:
                    end -= segment_end - self.token_len
                    segment_end = self.token_len
                try:
                    # print(f"Segment start: {segment_start}, Segment end: {segment_end}")
                    segment[segment_start:segment_end] = ecg_signal[channel_index][start:end]
                except:
                    return None, None
                qrs_segments.append(segment)
            
            channel_qrs_segments.append(qrs_segments)
            channel_pad_masks.append(pad_masks)
        return channel_qrs_segments, channel_pad_masks

    def assign_time_blocks(self, qrs_inds, interval_length=100):
        in_time = [(ind // interval_length) + 1 if ind !=0 else 0 for ind in qrs_inds]
        if len(in_time) < self.lead_len:
            for i in range(self.lead_len-len(in_time)):
                in_time.append(0)
        return in_time

    def qrs_to_sequence(self, channel_qrs_segments, qrs_inds,channel_pad_masks):
        qrs_sequence = []
        pad_mask = []
        in_chans = []
        in_times = []
        for channal_index, channel in enumerate(channel_qrs_segments):
            times = self.assign_time_blocks(qrs_inds[channal_index][:self.lead_len])
            in_times.extend(times)
            pad_mask.extend(channel_pad_masks[channal_index][:self.lead_len])
            # qrs_sequence.extend(channel[:self.lead_len])
            for i in range(self.lead_len):
                segments = channel[i]
            #     pad_mask.append(pad_mask[channal_index][i])
                qrs_sequence.append(segments)
            #     # in_chans.append(channal_index + 1)
                in_chans.append(self.used_chans[channal_index] + 1)
        return np.stack(qrs_sequence), np.array(in_chans), np.array(in_times),np.array(pad_mask)

    def forward(self, x ):
        x = x[self.used_chans, :]
        c, l = x.shape
        batch_qrs_seq = []
        batch_in_chans = []
        batch_in_times = []
        batch_mask_pad = []

    
        ecg_signal = x

        if ecg_signal.max() == 0:
            return None
        qrs_inds,flag = self.qrs_detection(ecg_signal)
        if len(qrs_inds[0]) == 0:
            return None
        channel_qrs_segments,channel_pad_masks = self.extract_qrs_segments(ecg_signal, qrs_inds)

        qrs_sequence, in_chans, in_times, pad_masks = self.qrs_to_sequence(
            channel_qrs_segments, qrs_inds,channel_pad_masks
        )

        batch_qrs_seq.append(qrs_sequence)
        batch_in_chans.append(in_chans)
        batch_in_times.append(in_times)
        batch_mask_pad.append(pad_masks)

        batch_qrs_seq = np.array(batch_qrs_seq).astype(np.float32)
        batch_in_chans = np.array(batch_in_chans)
        batch_in_times = np.array(batch_in_times)
        return batch_qrs_seq,batch_in_chans,batch_in_times


def read_dat(file_path:str=None):
    f = open(file_path, "rb")
    all_ecg = []
    for i in range(8):
        lead_data = []
        for pos in range(0, 500 * 10):
            b = f.read(2)
            value = struct.unpack("h", b)[0]
            lead_data.append(value)
        all_ecg.append(np.array(lead_data))
    f.close()    
    all_ecg.insert(2, all_ecg[1] - all_ecg[0])
    all_ecg.insert(3, -0.5 * (all_ecg[1] + all_ecg[0]))
    all_ecg.insert(4, all_ecg[0] - 0.5 * all_ecg[1])
    all_ecg.insert(5, all_ecg[1] - 0.5 * all_ecg[0])
    all_ecg = np.vstack(all_ecg)
    return all_ecg

def get_models():
    model = create_model('CLEAR_finetune_base',pretrained=False, num_classes=19, cls_token_num=12,padding_mask=False,atten_mask=False,mask_ratio=0)
    return model


used_channels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
Tokenizer = QRSTokenizer(
                fs=100,
                max_len=180,
                token_len=96,
                lead_len=15,
                used_chans=used_channels,
            )

checkpoint = torch.load('checkpoint-best.pth')

model = get_models()
model.load_state_dict(checkpoint,strict=False)
model.eval()

# data = read_dat('张巧妹_TJ30225239_2023-09-10081411.dat')
data = read_dat('47761422.dat')
data,in_chans,in_times = Tokenizer(data)
data,in_chans,in_times = torch.from_numpy(data),torch.from_numpy(in_chans),torch.from_numpy(in_times)

with torch.no_grad():
    output = model(data,in_chan_matrix=in_chans, in_time_matrix=in_times).detach().cpu().numpy()[0][:-1]
max_index = np.argmax(output)