import math
import torch
from timm.models import register_model
from torch import nn
from modeling_pretrain import ECGFM
import torch.nn.functional as F
from utils.moe import HierarchicalMoE

class WeightedSum(nn.Module):
    def __init__(self,input_dim =768):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (B, 12, 768)
        attn_scores = self.attn(x)           # (B, 12, 1)
        attn_weights = F.softmax(attn_scores, dim=1)  # (B, 12, 1)
        weighted_sum = (attn_weights * x).sum(dim=1, keepdim=True)  # (B, 1, 768)
        return weighted_sum[:,0,:]

class ECGFMClassifier(nn.Module):
    def __init__(
        self,
        seq_len,
        time_window,
        depth,
        embed_dim,
        heads,
        mlp_dim,
        dim_head=64,
        num_classes=None,
        dropout=0.0,
        emb_dropout=0.0,
        cls_token_num=1,
        attn_mask=False,
        padding_mask=False,
        mask_ratio=0.0,
        stage = 1
    ):
        super().__init__()
        self.padding_mask = padding_mask
        self.backbone = ECGFM(
            seq_len=seq_len,
            time_window=time_window,
            depth=depth,
            embed_dim=embed_dim,
            heads=heads,
            mlp_dim=mlp_dim,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout,
            cls_token_num=cls_token_num,
            padding_mask=padding_mask,
            attn_mask=attn_mask,
            mask_ratio=mask_ratio,
            stage=stage
        )
        self.stage = stage
        self.depth = depth
    
        if stage == 1:
            self.mlp_head = nn.Linear(embed_dim, num_classes)
        else:
            self.mlp_head = nn.Linear(embed_dim, num_classes)
            # self.mlp_head = nn.Linear(embed_dim, num_classes)
            # self.weight_sum = WeightedSum()
        # for i in range(self.depth):
        #     self.backbone.transformer.layers[i][0].adapter = FFTAdapter(768)

    def forward(
        self, x, in_chan_matrix=None, in_time_matrix=None, return_all_tokens=True,key_padding_mask=None,attn_mask=None,visual=False,visual_group=False
    ):  
        cls_token, mask = self.backbone(
            x,
            in_chan_matrix=in_chan_matrix,
            in_time_matrix=in_time_matrix,
            return_all_tokens=return_all_tokens,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            visual=visual
        )
        if self.stage==2 and visual_group:
            group_feature = torch.stack(cls_token,dim=1)
            pred = self.mlp_head(torch.mean(group_feature, dim=1))
            return group_feature,pred,self.mlp_head.weight

        if visual:
                return cls_token, mask
        if self.stage == 1:
            cls_token = cls_token[:,:12]
            cls_token = torch.mean(cls_token, dim=1)
        else:
            cls_token = torch.stack(cls_token,dim=1)
            cls_token = torch.mean(cls_token, dim=1)
        if visual:
            return cls_token
        output = self.mlp_head(cls_token)
        return output

    def get_num_layers(self):
        return self.depth

    @torch.jit.ignore
    def no_weight_decay(self):
        # return {"pos_embedding", "cls_token", "token_embed"}
        return {}

    def randomly_initialize_weights(self):
        for param in self.parameters():
            nn.init.uniform_(param, a=-0.1, b=0.1)


def get_model_default_params():
    return dict(
        seq_len=180,
        time_window=96,
        num_classes=1000,
        embed_dim=768,
        depth=8,
        heads=4,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

def get_model_large_default_params():
    return dict(
        seq_len=180,
        time_window=96,
        embed_dim=1024,
        depth=24,
        heads=16,
        dim_head=64,
        mlp_dim=2048,
        dropout=0.1,
        emb_dropout=0.1,
    )

@register_model
def CLEAR_finetune_base(pretrained=False, **kwargs):
    config = get_model_default_params()
    config["num_classes"] = kwargs["num_classes"]
    config["depth"] = 12
    config["heads"] = 8
    config["cls_token_num"] = kwargs["cls_token_num"]
    config["mlp_dim"] = 1024
    config["padding_mask"] = kwargs["padding_mask"]
    config["attn_mask"] = kwargs["atten_mask"]
    config['mask_ratio'] = kwargs['mask_ratio']
    model = ECGFMClassifier(**config)
    return model

@register_model
def CLEAR_finetune_large(pretrained=False, **kwargs):
    config = get_model_large_default_params()
    config["num_classes"] = kwargs["num_classes"]
    config["depth"] = 24
    config["heads"] = config["embed_dim"] // 64
    config["cls_token_num"] = kwargs["cls_token_num"]
    config["mlp_dim"] = 1024
    config["padding_mask"] = kwargs["padding_mask"]
    config["attn_mask"] = kwargs["atten_mask"]
    config['mask_ratio'] = kwargs['mask_ratio']
    model = ECGFMClassifier(**config)
    return model

@register_model
def CLEAR_HUG_finetune_large(pretrained=False, **kwargs):
    config = get_model_large_default_params()
    config["num_classes"] = kwargs["num_classes"]
    config["depth"] = 24
    config["heads"] = config["embed_dim"] // 64
    config["cls_token_num"] = kwargs["cls_token_num"]
    config["mlp_dim"] = 1024
    config["padding_mask"] = kwargs["padding_mask"]
    config["attn_mask"] = kwargs["atten_mask"]
    config['mask_ratio'] = kwargs['mask_ratio']
    config['stage'] = 2
    model = ECGFMClassifier(**config)
    return model

@register_model
def CLEAR_HUG_finetune_base(pretrained=False, **kwargs):
    config = get_model_default_params()
    config["num_classes"] = kwargs["num_classes"]
    config["depth"] = 12
    config["heads"] = 8
    config["cls_token_num"] = kwargs["cls_token_num"]
    config["mlp_dim"] = 1024
    config["padding_mask"] = kwargs["padding_mask"]
    config["attn_mask"] = kwargs["atten_mask"]
    config['mask_ratio'] = kwargs['mask_ratio']
    config['stage'] = 2
    model = ECGFMClassifier(**config)
    return model


if __name__ == "__main__":
    model = CLEAR_finetune_base(num_classes=12)
    print(model)
