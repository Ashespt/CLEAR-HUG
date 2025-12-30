from typing import Tuple
import math
import torch
from timm import create_model
from timm.models import register_model
from torch import nn
from einops import rearrange, repeat, pack
from matplotlib import pyplot as plt
import numpy as np
from utils.mask_atten import MaskedTransformerEncoder
from utils.pos_embed import get_2d_sincos_pos_embed
import torch.nn.functional as F
from utils.moe import HierarchicalMoE
from utils.augmentation import feature_aug
def random_masking(x, mask_ratio):
    """
    intput x: [bs, seq_len, embed_dim] ; mask_ratio: float
    output mask matrix: (bool)[bs, seq_len]
    """
    bs, seq_len, dim = x.shape
    len_keep = int(seq_len * (1 - mask_ratio))
    noise = torch.rand(bs, seq_len, device=x.device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    mask = torch.ones([bs, seq_len], device=x.device)
    mask[:, :len_keep] = 0
    mask = torch.gather(mask, dim=1, index=ids_restore)
    return mask.to(torch.bool)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, embed_dim, heads=8, head_dim=64, dropout=0.0):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == embed_dim)

        self.heads = heads
        self.scale = head_dim**-0.5

        self.norm = nn.LayerNorm(embed_dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(embed_dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, embed_dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, embed_dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            embed_dim=embed_dim,
                            heads=heads,
                            head_dim=dim_head,
                            dropout=dropout,
                        ),
                        FeedForward(embed_dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class SpatialEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(SpatialEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, in_chan_matrix):
        spatial_embeddings = self.embed(
            in_chan_matrix
        )  # [batch_size, seq_len, embed_dim]
        return x + spatial_embeddings


class TemporalEmbedding(nn.Module):
    def __init__(self, num_embeddings, embed_dim):
        super(TemporalEmbedding, self).__init__()
        self.embed = nn.Embedding(num_embeddings, embed_dim)

    def forward(self, x, time_index_matrix):
        temporal_embeddings = self.embed(
            time_index_matrix
        )  # [batch_size, seq_len, embed_dim]
        return x + temporal_embeddings

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

class ECGFM(nn.Module):
    def __init__(
        self,
        seq_len,
        time_window,
        depth,
        embed_dim,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
        decoder_depth=4, #8
        decoder_num_heads=16,
        decoder_embed_dim=512,
        cls_token_num=12,
        norm_pix_loss=True,
        padding_mask=False,
        attn_mask=False,
        stage=1,
        mask_ratio=0.8
    ):
        super().__init__()
        self.padding_mask = padding_mask
        self.attn_mask =attn_mask
        self.time_window = time_window
        self.mask_ratio = mask_ratio
        self.T = 0.07
        self.stage = stage
        self.token_embed = TokenEmbedding(c_in=time_window, d_model=embed_dim)
        self.cls_token_num = cls_token_num
        self.cls_token = nn.Parameter(torch.randn(self.cls_token_num,embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.decoder_num_heads = decoder_num_heads
        self.norm_pix_loss = norm_pix_loss
        # learnable position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len + self.cls_token_num, embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, seq_len + self.cls_token_num, decoder_embed_dim))

        self.spa_embed = SpatialEmbedding(num_embeddings=16, embed_dim=embed_dim)
        self.tem_embed = TemporalEmbedding(num_embeddings=16, embed_dim=embed_dim)

        self.dropout = nn.Dropout(emb_dropout)
        self.encoder_transformer = MaskedTransformerEncoder(
            embed_dim=embed_dim,
            depth=depth,
            heads=heads,
            # dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.decoder_transformer = MaskedTransformerEncoder(
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            heads=decoder_num_heads,
            # dim_head=decoder_embed_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
        )
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        if stage == 2:
            self.moe = HierarchicalMoE(input_dim=embed_dim)
        self.norm_layer = nn.LayerNorm(embed_dim)
        self.initialize_weights(decoder_embed_dim,time_window)

    def initialize_weights(self,decoder_embed_dim,time_window):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 180, cls_token=True,cls_token_num=self.cls_token_num)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], 180, cls_token=True,cls_token_num=self.cls_token_num)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, time_window, bias=True)


    def random_masking_atten(self, x, mask_ratio):
        N, L, D = x.shape

        num_tokens_plead = L//self.cls_token_num

        assert num_tokens_plead * self.cls_token_num == L

        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(1, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep].repeat((N,1))
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([1, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        attn_mask = torch.zeros([1, (L+self.cls_token_num), (L+self.cls_token_num)]) #初始化mask
        attn_mask[:, :self.cls_token_num, :self.cls_token_num] = torch.eye(self.cls_token_num).int().clone() # cls token和cls token关系
        cls_mask = torch.zeros([1, self.cls_token_num, L])
        cls_mask_r = torch.zeros([1, L, self.cls_token_num])
        for i in range(self.cls_token_num): # cls token 与 heatbeat tokens 之间的关系, 0 not attend, 1 attend
            cls_mask[:, i, num_tokens_plead * i:num_tokens_plead * (i + 1)] = 1 # should be 1
            cls_mask_r[:, num_tokens_plead * i:num_tokens_plead * (i + 1), i] = 1 # should be 1
        attn_mask[:, :self.cls_token_num, self.cls_token_num:] = cls_mask
        attn_mask[:, self.cls_token_num:, :self.cls_token_num] = cls_mask_r
        for n in range(1): # heatbeat tokens 与 heatbeat tokens 之间的关系, 0 not attend, 1 attend
            mask_idx_copy = torch.where(mask[n] == 1)[0]
            mask_idx = list(mask_idx_copy)
            for l_src in range(L):
                mask_line = torch.zeros(L)
                lead_id = l_src // num_tokens_plead
                if l_src in mask_idx:
                    mask_line[l_src % num_tokens_plead::num_tokens_plead] = 1  # should be 1
                else:
                    mask_line[:] = 1
                    mask_line[mask_idx_copy] = 0

                attn_mask[n, l_src + self.cls_token_num, self.cls_token_num:] = mask_line
        
        # mask = mask.repeat(N, 1)
        attn_mask = attn_mask.repeat(N, 1, 1)

        # encoder cls mask
        numunmask_lead = (1-mask.reshape(1,self.cls_token_num,int(L/self.cls_token_num)).repeat((N,1,1))).sum(dim=-1)
        attn_mask_encoder = torch.zeros(N,len_keep+self.cls_token_num,len_keep+self.cls_token_num)
        
        for j in range(self.cls_token_num):
            vec = torch.ones(len_keep+self.cls_token_num) # 1 not attend, 0 attend
            vec[int(numunmask_lead[0,:j].sum()):int(numunmask_lead[0,:(j+1)].sum())] = 0
            attn_mask_encoder[:,j,:] = vec

        # return x_masked, ~attn_mask.to(torch.bool).cuda(), mask.repeat((N,1)), ids_restore.repeat((N,1)), attn_mask_encoder.cuda()
        return x_masked, ~attn_mask.to(torch.bool).cuda(), mask, ids_restore.repeat((N,1)), None

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore

    def forward_feature(
        self, x, mask_bool_matrix=None, key_padding_mask=None, in_chan_matrix=None, in_time_matrix=None,attn_mask=None
    ): 
        """
        mask_bool_matrix: 1: mask, 0: unmask
        """
        x = self.token_embed(x)
        b, seq_len, embed_dim = x.shape
        # position, spatial, temporal
        if in_chan_matrix is not None:
            x = self.spa_embed(x, in_chan_matrix)

        if in_time_matrix is not None:
            x = self.tem_embed(x, in_time_matrix)
        x += self.pos_embed[:,self.cls_token_num:,:]
        
        x, attn_mask, mask, ids_restore, attn_mask_encoder = self.random_masking_atten(x, self.mask_ratio)
        # else:
        #     x, mask, ids_restore = self.random_masking_lead_and_token(x)
        cls_tokens = repeat(self.cls_token, "c d -> b c d", b=b)+ self.pos_embed[:, :self.cls_token_num, :]
        x, ps = pack([cls_tokens, x], "b * d")
        

        # if mask_bool_matrix is None:
        #     mask_bool_matrix = torch.zeros((b, seq_len), dtype=torch.bool).to(x.device)

        # mask_tokens = self.mask_token.expand(b, seq_len, -1)
        # w = mask_bool_matrix.unsqueeze(-1).type_as(mask_tokens)
        # x[:,self.cls_token_num:,:] = x[:,self.cls_token_num:,:] * (1 - w) + mask_tokens * w

        # x = self.dropout(x)
        x = self.encoder_transformer(x,key_padding_mask=key_padding_mask,attn_mask=attn_mask_encoder)
        x = self.norm_layer(x)
        if self.stage == 1:
            return x, attn_mask, mask,ids_restore
        else:
            return x, None, None, None

    def forward_decoder(self, x, key_padding_mask, attn_mask=None,ids_restore=None,return_all_tokens=False):
        x = self.decoder_embed(x)
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x[:, self.cls_token_num:,:].shape[1], 1)
        x_ = torch.cat([x[:, self.cls_token_num:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :self.cls_token_num, :], x_], dim=1)
        x = x + self.decoder_pos_embed
        x = self.decoder_transformer(x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
        x = self.decoder_norm(x)
        x = self.decoder_pred(x)
        if return_all_tokens:
            return x
        return x[:,self.cls_token_num:,:]

    def forward_loss(self, target, pred, mask):
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_cl(self, output,criterion):
        B = output.shape[0]
        x,x1 = output[:B//2],output[B//2:]
        sim = torch.einsum("nc,mc->nm", [x, x1])
        l_pos = sim.diagonal()
        # negative logits: NxK
        mask = ~torch.eye(B//2, dtype=torch.bool)
        l_neg = sim[mask].reshape(B//2, B//2 - 1)
        logits = torch.cat([l_pos[:,None], l_neg], dim=1)
        logits /= self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = criterion(logits, labels)
        return loss

    def forward(
        self,
        signals,
        mask_bool_matrix=None,
        key_padding_mask=None,
        in_chan_matrix=None,
        in_time_matrix=None,
        attn_mask=None,
        return_all_tokens=False,
        return_qrs_tokens=False,
        mask_ratio=0.8,
        criterion=None,
        visual=False
    ):  
        if self.norm_pix_loss:
            mean = signals.mean(dim=-1, keepdim=True)
            var = signals.var(dim=-1, keepdim=True)
            signals = (signals - mean) / (var + 1.e-6)**.5
        if not self.padding_mask:
            key_padding_mask = None
        if not self.attn_mask:
            attn_mask = None
        if key_padding_mask is not None:
            insert_padding_mask = torch.zeros((key_padding_mask.shape[0], self.cls_token_num), dtype=torch.bool).cuda()
            key_padding_mask = torch.cat([insert_padding_mask,key_padding_mask], dim=1)
        # x,mask = self.forward_feature(signals, mask_bool_matrix, key_padding_mask, in_chan_matrix, in_time_matrix,attn_mask) # bs, 192, 768
        if self.stage == 1:
            x,attn_mask,mask,ids_restore = self.forward_feature(signals, mask_bool_matrix, key_padding_mask, in_chan_matrix, in_time_matrix) # bs, 192, 768
        else:
            x,attn_mask,mask,ids_restore = self.forward_feature(signals, None, None, in_chan_matrix, in_time_matrix) # bs, 192, 768
        if return_all_tokens and not visual:
            if self.stage == 1:
                return x,None
            else:
                outputs = self.moe(x[:,:12,:])
                return outputs,None

        if return_qrs_tokens:
            return x[:, self.cls_token_num:],None

        if self.stage == 1:
            pred = self.forward_decoder(x,key_padding_mask,attn_mask,ids_restore,return_all_tokens=return_all_tokens)
            if visual:
                return pred,mask
            loss_rec = self.forward_loss(signals,pred,mask)
            return loss_rec,None
        else:
            B = x.shape[0]
            # x[B//2:,:,:] = feature_aug(x[B//2:,:,:])
            x[:B//2,:,:] = feature_aug(x[:B//2,:,:])
            outputs = self.moe(x[:,:12,:]) # B, 7, 768
            if visual:
                return outputs, None
            loss_cl = 0
            for output in outputs:
                loss_cl+= self.forward_loss_cl(output,criterion)
            return loss_cl/7,None

def get_model_default_params():
    return dict(
        seq_len=180,
        time_window=96,
        embed_dim=768,
        depth=12,
        heads=12,
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
def CLEAR(pretrained=False, **kwargs):
    config = get_model_default_params()

    config["depth"] = 12
    config["heads"] = 8

    config["mlp_dim"] = 1024
    config["padding_mask"] = kwargs["padding_mask"]
    config["attn_mask"] = kwargs["atten_mask"]
    config["cls_token_num"] = kwargs["cls_token_num"]
    config["stage"] = 1

    model = ECGFM(**config)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def CLEAR_large(pretrained=False, **kwargs):
    config = get_model_large_default_params()

    config["depth"] = 24
    config["heads"] = config["embed_dim"] // 64

    # config["decoder_depth"]=6
    # config["decoder_num_heads"]=8
    config["decoder_depth"]=24
    config["decoder_num_heads"]=16
    config["decoder_embed_dim"]=1024
    config["mlp_dim"] = 1024
    config["padding_mask"] = kwargs["padding_mask"]
    config["attn_mask"] = kwargs["atten_mask"]
    config["cls_token_num"] = kwargs["cls_token_num"]
    config["stage"] = 1

    model = ECGFM(**config)
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model

