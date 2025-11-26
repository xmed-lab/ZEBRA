import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from timm.models.vision_transformer import Block as TransBlock



def weights_init(m):
    classname = m.__class__.__name__
    if 'Linear' in classname or 'Embedding' == classname:
        # print("Initializing Module {classname}.")
        nn.init.trunc_normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            # if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    if 'LayerNorm' in classname:
        nn.init.constant_(m.weight, 1.0)
        nn.init.constant_(m.bias, 0.0)


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)
    # return F.relu(x)


def get_norm_layer(in_channels, mode):
    if mode == 'bn':
        return nn.BatchNorm2d(in_channels)
    elif mode == 'in':
        return nn.InstanceNorm2d(in_channels, affine=False)
    elif mode == 'gn':
        return nn.GroupNorm(num_groups=in_channels // 32, num_channels=in_channels)
    elif mode == 'none':
        return nn.Identity()
    else:
        raise ValueError


def get_1d_sincos_pos_embed(embed_dim, length, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_l = np.arange(length, dtype=np.float32)

    grid_l = grid_l.reshape([1, length])
    pos_embed = get_1d_sincos_pos_embed_from_grid(embed_dim, grid_l)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed(embed_dim, grid_size=(), cls_token=False, cond_length=1):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([cond_length, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class Block(nn.Module):
    """
    Transformer encoder using MultiHeadAttention and MLP along with skip connections and LayerNorm
    """

    def __init__(self, config):
        super(Block, self).__init__()
        drop_p = config['drop_rate']
        emb_dim = config['embed_dim']
        num_heads = config['num_heads']

        self.attention = nn.MultiheadAttention(emb_dim, num_heads=num_heads, dropout=drop_p)
        self.ln1 = nn.LayerNorm(emb_dim, eps=1e-12)
        self.ln2 = nn.LayerNorm(emb_dim, eps=1e-12)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(p=drop_p)
        )
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, x, mask=None):
        # x: [B, L, C]
        key = x.transpose(0, 1)  # [L, B, C]
        attn, _ = self.attention(key, key, key, need_weights=False, attn_mask=mask)  # [L, B, C]
        attn = self.dropout(attn.transpose(0, 1))
        x = self.ln1(x + attn)
        mlp = self.mlp(x)
        x = self.ln2(x + mlp)

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, ndim, nhead, dropout):
        super().__init__()
        assert ndim % nhead == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(ndim, ndim)
        self.query = nn.Linear(ndim, ndim)
        self.value = nn.Linear(ndim, ndim)
        # regularization
        self.drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(ndim, ndim)
        self.n_head = nhead

    def forward(self, q, k, v, mask=None):
        # v: [B, L, C]
        # mask: [B, 1, Lq, Lk]
        B, Lq, C = q.size()
        Lk = k.size(1)
        Lv = v.size(1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(k).view(B, Lk, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lk, hs)
        q = self.query(q).view(B, Lq, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lq, hs)
        v = self.value(v).view(B, Lv, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lv, hs)

        # causal self-attention; Self-attend: (B, nh, Lq, hs) x (B, nh, hs, Lk) -> (B, nh, Lq, Lk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            # mask:[B, 1, Lq, Lk]
            assert mask.dim() == 4
            att = att.masked_fill(mask == 0, float('-inf'))

        if v.dtype == torch.float16:
            att = att.to(torch.float32)
            fp16 = True
        else:
            fp16 = False
        att = F.softmax(att, dim=-1)  # (B, nh, Lq, kL*)
        if fp16:
            att = att.to(torch.float16)
        att = self.drop(att)
        y = att @ v  # (B, nh, Lq, Lk*) x (B, nh, Lv, hs) -> (B, nh, Lq, hs)

        y = y.transpose(1, 2).contiguous().view(B, Lq, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class TransSelfEncoder(nn.Module):
    def __init__(self, ndim, nhead, dropout, norm_before):
        super().__init__()
        # ndim = config['embed_dim']
        # nhead = config['num_heads']
        # dropout = config['drop_rate']
        # norm_before = config['norm_before']

        self.self_attn = MultiHeadAttention(ndim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(ndim, ndim * 4)
        self.linear2 = nn.Linear(ndim * 4, ndim)

        self.norm1 = nn.LayerNorm(ndim, eps=1e-12)
        self.norm2 = nn.LayerNorm(ndim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm_before = norm_before

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, src, src_pos, mask=None):
        # src: [B, L, C]
        # src_pos: [B, L, C]
        # mask: [B, 1, L, L]
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, src_pos)
        src2 = self.self_attn(q=q, k=k, v=src2, mask=mask)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)

        src2 = self.linear2(self.dropout(self.gelu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward_post(self, src, src_pos, mask=None):
        # src: [B, L, C]
        # src_pos: [B, L, C]
        # mask: [B, 1, L, L]
        q = k = self.with_pos_embed(src, src_pos)
        src2 = self.self_attn(q=q, k=k, v=src, mask=mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, src_pos, mask=None):
        if self.norm_before:
            return self.forward_pre(src, src_pos, mask)
        else:
            return self.forward_post(src, src_pos, mask)


class TransCrossDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        ndim = config['embed_dim']
        nhead = config['num_heads']
        dropout = config['drop_rate']
        norm_before = config['norm_before']

        self.self_attn = MultiHeadAttention(ndim, nhead, dropout=dropout)
        self.cross_attn = MultiHeadAttention(ndim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(ndim, ndim * 4)
        self.linear2 = nn.Linear(ndim * 4, ndim)

        self.norm1 = nn.LayerNorm(ndim, eps=1e-12)
        self.norm2 = nn.LayerNorm(ndim, eps=1e-12)
        self.norm3 = nn.LayerNorm(ndim, eps=1e-12)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm_before = norm_before

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward_pre(self, tgt, tgt_pos, mem, mem_pos, mask1=None, mask2=None):
        # tgt: [B, L, C]
        # tgt_pos: [B, L, C]
        # mem: [B, M, C]
        # mem_pos: [B, M, C]
        # mask: [B, 1, L, L]
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, tgt_pos)
        tgt2 = self.self_attn(q=q, k=k, v=tgt2, mask=mask1)
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.cross_attn(q=self.with_pos_embed(tgt2, tgt_pos),
                               k=self.with_pos_embed(mem, mem_pos),
                               v=mem, mask=mask2)

        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.gelu(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward_post(self, tgt, tgt_pos, mem, mem_pos, mask1=None, mask2=None):
        # tgt: [B, L, C]
        # tgt_pos: [B, L, C]
        # mem: [B, M, C]
        # mem_pos: [B, M, C]
        # mask: [B, 1, L, L]
        q = k = self.with_pos_embed(tgt, tgt_pos)
        tgt2 = self.self_attn(q=q, k=k, v=tgt, mask=mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.cross_attn(q=self.with_pos_embed(tgt, tgt_pos),
                               k=self.with_pos_embed(mem, mem_pos),
                               v=mem, mask=mask)

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.gelu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward(self, tgt, tgt_pos, mem, mem_pos, mask1=None, mask2=None):
        if self.norm_before:
            return self.forward_pre(tgt, tgt_pos, mem, mem_pos, mask1, mask2)
        else:
            return self.forward_post(tgt, tgt_pos, mem, mem_pos, mask1, mask2)


class MultiHeadDotProductAttention(nn.Module):
    def __init__(self, ndim, nhead, dropout):
        super().__init__()
        assert ndim % nhead == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(ndim, ndim)
        self.query = nn.Linear(ndim, ndim)
        self.value = nn.Linear(ndim, ndim)
        # regularization
        self.drop = nn.Dropout(dropout)
        # output projection
        self.proj = nn.Linear(ndim, ndim)
        self.n_head = nhead

    def forward(self, q, k, v, mask=None):
        # v: [B, L, C]
        # mask: [B, 1, Lq, Lk]
        B, Lq, C = q.size()
        Lk = k.size(1)
        Lv = v.size(1)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(k).view(B, Lk, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lk, hs)
        q = self.query(q).view(B, Lq, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lq, hs)
        v = self.value(v).view(B, Lv, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, Lv, hs)

        # causal self-attention; Self-attend: (B, nh, Lq, hs) x (B, nh, hs, Lk) -> (B, nh, Lq, Lk)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            # mask:[B, 1, Lq, Lk]
            assert mask.dim() == 4
            att = att.masked_fill(mask == 0, float('-inf'))

        if v.dtype == torch.float16:
            att = att.to(torch.float32)
            fp16 = True
        else:
            fp16 = False
        att = F.softmax(att, dim=-1)  # (B, nh, Lq, kL*)
        if fp16:
            att = att.to(torch.float16)
        att = self.drop(att)
        y = att @ v  # (B, nh, Lq, Lk*) x (B, nh, Lv, hs) -> (B, nh, Lq, hs)

        y = y.transpose(1, 2).contiguous().view(B, Lq, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.proj(y)
        return y


class SelfAttention(nn.Module):
    def __init__(self, ndim, nhead, dropout):
        super().__init__()
        # ndim = config['embed_dim']
        # nhead = config['num_heads']
        # dropout = config['drop_rate']

        self.self_attn = MultiHeadDotProductAttention(ndim, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.gelu = nn.GELU()
        self.linear1 = nn.Linear(ndim, ndim * 4)
        self.linear2 = nn.Linear(ndim * 4, ndim)

        self.norm1 = nn.LayerNorm(ndim, eps=1e-12)
        self.norm2 = nn.LayerNorm(ndim, eps=1e-12)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src, src_pos, mask=None):
        # src: [B, L, C]
        # src_pos: [B, L, C]
        # mask: [B, 1, L, L]
        q = k = self.with_pos_embed(src, src_pos)
        src2 = self.self_attn(q=q, k=k, v=src, mask=mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.gelu(self.linear1(src)))
        src = src + self.dropout2(src2)

        return self.norm2(src)


# condition embedding
class PatchEmbed1D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """

    def __init__(self, num_voxels=224, patch_size=16, in_chans=1, embed_dim=768):
        super().__init__()
        num_patches = num_voxels // patch_size
        self.patch_shape = patch_size
        self.num_voxels = num_voxels
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, K = x.shape  # batch, channel, voxels
        # assert K == self.num_voxels, \
        #     f"Input fmri length ({K}) doesn't match model ({self.num_voxels})."
        x = self.proj(x).transpose(1, 2).contiguous()  # put embed_dim at the last dimension
        return x  # [B, K, C]


class PatchEmbed2D(nn.Module):
    """ 1 Dimensional version of data (fmri voxels) to Patch Embedding
    """

    def __init__(self, img_size=(224, 224), patch_size=16, in_chans=1, embed_dim=768, flatten=True, bias=True):
        super().__init__()
        assert isinstance(img_size, tuple)
        self.img_size = img_size
        self.patch_size = patch_size
        self.flatten = flatten
        assert img_size[0] % patch_size == 0
        assert img_size[1] % patch_size == 0

        self.num_h = img_size[0] // patch_size
        self.num_w = img_size[1] // patch_size
        self.num_patches = self.num_h * self.num_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape  # batch, channel, voxels
        assert H == self.img_size[0]
        assert W == self.img_size[1]

        x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2).contiguous()  # NCHW -> NLC

        return x  # [B, L, C]


class PoseMLP(nn.Module):
    def __init__(self, embedding_dim):
        super(PoseMLP, self).__init__()
        self.block1 = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # x B*13*3
        x = self.block1(x)
        x = self.block2(x)
        out = self.block3(x)
        return out


class PoseConv(nn.Module):
    def __init__(self, embedding_dim):
        super(PoseConv, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(inplace=True)
            nn.GELU()
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(inplace=True)
            nn.GELU()
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(inplace=True)
            nn.GELU()
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, embedding_dim, kernel_size=3, stride=2, padding=1),
            # nn.ReLU(inplace=True)
            nn.GELU()
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)  # [B, C, H, W]

        out = x.flatten(start_dim=2).transpose(1, 2)  # [B, HW, C]

        return out


class CoordPose(nn.Module):
    def __init__(self, config):
        super(CoordPose, self).__init__()

        feat_dim = config['num_part']
        norm_mode = config['norm_mode']

        # downsampling
        self.conv1 = nn.Conv2d(feat_dim + 2, feat_dim, kernel_size=1, stride=1, padding=0)
        self.norm1 = get_norm_layer(feat_dim, norm_mode)

        self.conv2 = nn.Conv2d(feat_dim + 2, feat_dim, kernel_size=1, stride=1, padding=0)
        self.norm2 = get_norm_layer(feat_dim, norm_mode)

        self.mask = nn.Conv2d(feat_dim, 1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

        self.flow = nn.Conv2d(feat_dim, 2, kernel_size=1, stride=1, padding=0)
        self.tanh = nn.Tanh()

    def get_coord_map(self, x, add_dist=False):
        '''
        :param x: [B, C, H, W]
        :return: [B, 2(3), H, W]
        '''
        batch_size, _, h, w = x.size()
        rows = torch.arange(0, h).view(1, 1, h, 1).float().repeat_interleave(w, dim=3) / (h - 1)
        cols = torch.arange(0, w).view(1, 1, 1, w).float().repeat_interleave(h, dim=2) / (w - 1)

        rows = rows.repeat_interleave(batch_size, dim=0)
        cols = cols.repeat_interleave(batch_size, dim=0)

        coords = torch.cat((rows, cols), dim=1)  # [B, 2, H, W]
        coords = coords * 2. - 1.  # ranges -1 ~ 1
        if add_dist:
            dist_map = torch.sqrt(torch.pow(coords[:, 0], 2) + torch.pow(coords[:, 1], 2))
            coords = torch.cat((coords, dist_map.unsqueeze(1)), dim=1)  # [B, 3, H, W]

        return coords.to(x.device)

    def forward(self, src_mask, tgt_mask):
        '''
        :param src_mask: [B, K, H, W]
        :param tgt_mask: [B, K, H, W]
        :return: [B, 2, H, W], [B, 1, H, W]
        '''

        diff = tgt_mask - src_mask

        coord = self.get_coord_map(diff)
        diff2 = torch.cat([diff, coord], dim=1)
        diff2 = self.conv1(diff2)
        diff = nonlinearity(self.norm1(diff2))

        coord = self.get_coord_map(diff)
        diff2 = torch.cat([diff, coord], dim=1)
        diff2 = self.conv2(diff2)
        diff = nonlinearity(self.norm2(diff2))

        mask = self.sigmoid(self.mask(diff))
        flow = self.tanh(self.flow(diff))

        return flow, mask


class MaskGIT_Image2RGB(nn.Module):
    def __init__(self, config):
        super(MaskGIT_Image2RGB, self).__init__()
        self.config = config.Model
        decoder_embed_dim = config.Model['decoder_embed_dim']
        self.num_voxel = config.Data['num_voxel']
        num_latent_size = config.Model['resolution'] // config.Model['patch_size']
        self.num_embed = config.Model['num_codebook']  # + mask_token    !!!!!!!!!!!!!
        num_seq_length = num_latent_size ** 2 + 1
        img_dim = 1024  # 1024 / 1280

        # decoder
        self.decoder_embed = nn.Linear(img_dim, decoder_embed_dim, bias=True)
        self.token_emb = nn.Embedding(self.num_embed + 1000 + 1, decoder_embed_dim)  # [2025, 768]
        self.de_pos_emb = nn.Embedding(num_seq_length, decoder_embed_dim)  # [257, 768]
        self.ln = nn.LayerNorm(decoder_embed_dim)  # [768]
        self.dropout2 = nn.Dropout(config.Model['drop_rate'])
        self.decoder = nn.ModuleList(
            [SelfAttention(decoder_embed_dim, config.Model['num_heads'], config.Model['drop_rate'])
             for _ in range(config.Model['decoder_depth'])])

        self.mlps = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.LayerNorm(decoder_embed_dim, eps=1e-12)
        )
        self.mlps_bias = nn.Parameter(torch.zeros(self.num_embed + 1000 + 1))  # [2025]
        self.apply(weights_init)

    def forward(self, x, cls):
        # x: [B, L]
        # cls: [B, C]

        # embed tokens
        c = self.decoder_embed(cls[:, None])  # [B, 1, C]
        z = self.token_emb(x)  # [B, L, C]
        z = torch.cat([c, z], dim=1)  # [B, 1+L, C]
        z_pos = self.de_pos_emb(torch.arange(z.shape[1], dtype=torch.long, device=c.device)[None])  # [1, 1+L, C]
        z = self.dropout2(self.ln(z + z_pos))  # [B, 1+L, C]

        # apply Transformer blocks
        for blk in self.decoder:
            z = blk(z, None)  # [B, 1+L, C]

        # pred
        emb = self.mlps(z)  # [B, 1+L, C]
        logits = torch.matmul(emb, self.token_emb.weight.T) + self.mlps_bias  # [B, 1+L, N+1000+1]

        return logits[:, 1:, :self.num_embed].contiguous()  # [B, L, N]


class fMRI_MLP(nn.Module):
    def __init__(self, config):
        super(fMRI_MLP, self).__init__()
        embed_dim = config.Model['embed_dim']
        num_head = config.Model['num_heads']
        drop_p = config.Model['drop_rate']
        # in_chans = config.Model['in_chans']
        img_dim = config.Model['image_dim']

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 257 + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            TransBlock(embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True,
                       proj_drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config.Model['encoder_depth'])])
        self.norm = nn.LayerNorm(embed_dim)
        self.pred = nn.Linear(embed_dim, img_dim, bias=True)

        self.initialize_weights()

    def initialize_weights(self, ):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_1d_sincos_pos_embed(self.pos_embed.shape[-1], 257, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        # x: [B, 1, L]
        # add pos embed w/o cls token

        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        z = self.pred(x[:, :1])
        return z

    def forward(self, fmri):
        # fmri: [B, 1, K]

        preds = self.forward_encoder(fmri)
        return preds



class CrossAttention(nn.Module):
    def __init__(self, emb_dim, attention_dropout_rate=0.1):
        super().__init__()

        self.q = nn.Linear(emb_dim, emb_dim, bias=False)
        self.k = nn.Linear(emb_dim, emb_dim, bias=False)
        self.v = nn.Linear(emb_dim, emb_dim, bias=False)

        self.softmax = nn.Softmax(dim=-1)

        self.out = nn.Linear(emb_dim, emb_dim, bias=False)
        self.attn_dropout = nn.Dropout(attention_dropout_rate)
        self.proj_dropout = nn.Dropout(attention_dropout_rate)


    def forward(self, q_feat, kv_feat):
        q = self.q(q_feat)
        k = self.k(kv_feat)
        v = self.v(kv_feat)

        scale = q_feat.shape[-1] ** -0.5

        cross_attn = torch.matmul(q, k.transpose(-1, -2))

        cross_attn = self.attn_dropout(self.softmax(cross_attn * scale))
        cross_attn = torch.matmul(cross_attn, v)

        out = self.out(cross_attn)
        out = self.proj_dropout(out)

        return out



class fMRI_Autoencoder(nn.Module):
    def __init__(self, config):
        super(fMRI_Autoencoder, self).__init__()
        patch_size = config.Data['patch_size']
        image_size = tuple(config.Data['image_size'])

        # num_voxel = config.Data['num_voxel']
        embed_dim = config.Model['embed_dim']
        decoder_embed_dim = config.Model['decoder_embed_dim']
        num_head = config.Model['num_heads']
        drop_p = config.Model['drop_rate']
        in_chans = config.Model['in_chans']
        img_dim = 2048

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed2D(image_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            TransBlock(embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True,
                       drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config.Model['encoder_depth'])])
        self.norm = nn.LayerNorm(embed_dim)

        self.pred = nn.Linear(embed_dim, img_dim, bias=True)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(img_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            TransBlock(decoder_embed_dim, num_head, mlp_ratio=4.0, qkv_bias=True,
                       drop=drop_p, attn_drop=drop_p, drop_path=drop_p, norm_layer=nn.LayerNorm)
            for _ in range(config.Model['decoder_depth'])])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)

        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)

        # --------------------------------------------------------------------------
        self.initialize_weights()
        self.in_chans = in_chans

    def initialize_weights(self, ):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], (self.patch_embed.num_h, self.patch_embed.num_w),
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    (self.patch_embed.num_h, self.patch_embed.num_w), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            torch.nn.init.normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size
        h = self.patch_embed.num_h
        w = self.patch_embed.num_w
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, self.in_chans))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p, w * p))
        return imgs

    def forward_encoder(self, x):
        # x: [B, C, H ,W]
        x = self.patch_embed(x)  # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        brain_rep = x[:, 1:]

        return brain_rep

    def forward_encoder_z(self, x):
        # x: [B, C, H ,W]
        x = self.patch_embed(x)  # [B, K, C]

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        brain_rep = x[:, 1:]
        cls = x[:, :1]
        z = self.pred(cls)

        return brain_rep, z


    def forward_decoder(self, x):
        # x: [B, 1, C]
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], self.patch_embed.num_patches, 1)
        z = torch.cat([x, mask_tokens], dim=1)  # append cls token

        # add pos embed
        z = z + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            z = blk(z)  # [B, 1+K, C]
        z = self.decoder_norm(z)  # [B, 1+K, C]

        # predictor projection
        out = self.decoder_pred(z)  # [B, 1+K, p*p*3]
        return out[:, 1:, :]  # [B, K, p*p*3]

    def forward(self, fmri, mask=None):
        # fmri: [B, 1, K]

        latent = self.forward_encoder(fmri)
        rec = self.forward_decoder(latent)
        loss = self.calculate_loss(fmri, rec, mask=mask)

        return rec, loss

    def calculate_loss(self, target, pred, mask=None):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, H, W], 0 is keep, 1 is remove,
        """
        pred = self.unpatchify(pred)  # [N, 1, H, W]

        loss = (pred - target) ** 2  # [N, 1, H, W]
        loss = loss.flatten(start_dim=1)

        if mask is not None:
            mask = mask[:, None].repeat(1, pred.shape[1], 1, 1).flatten(start_dim=1)  # [N, L]
            loss = (loss * mask).sum(-1) / mask.sum(-1)

        else:
            loss = loss.mean(-1)

        # return loss.mean()
        return loss


