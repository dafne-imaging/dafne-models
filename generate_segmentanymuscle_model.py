#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Copyright (c) 2026 Dafne-Imaging Team
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
Dafne plugin wrapping SegmentAnyMuscle (https://github.com/mazurowski-lab/SegmentAnyMuscle),
a fine-tuned Segment Anything Model for automatic (promptless) muscle segmentation in MRI/CT
volumes.

The underlying network is MobileSAM's TinyViT image encoder + SAM's mask decoder, with
encoder and mask-decoder adapter layers and a mixture-of-experts (MoE) routing mechanism
(moe=10, k=10) added on top, fine-tuned end to end. Only a handful of classes from the
upstream repo's sam/modeling/ package are actually exercised by the shipped checkpoint's
configuration (adapters on, LoRA/LST off, `vit_t`/TinyViT backbone only), and SegmentAnyMuscle
itself only *modified* some of them -- LayerNorm2d/MLPBlock (common.py), PromptEncoder (for
the promptless points=None/boxes=None/masks=None path this plugin uses), the two-way
Attention block, and mask_decoder's hypernetwork MLP are all untouched copies of Meta's
original segment-anything, so init_model_function below imports those straight from the
`segment-anything-py` PyPI package instead of re-embedding them. Only the classes
SegmentAnyMuscle actually changed -- Adapter/moe_forward, TinyViT (MobileSAM's encoder, not
part of segment-anything at all), MaskDecoder (needs an extra `training` passthrough), and
TwoWayTransformer/TwoWayAttentionBlock (carry the adapter/MoE logic) -- are defined locally,
trimmed of the unused LoRA/LST/multi-GPU-split/ViT-B/L/H code paths and of classes
SegmentAnyMuscle's own build_sam_vit_t() never instantiates (PromptEncoder's
auto_cls_emb/attention_fusion/PromptAutoEncoder, MaskDecoder's SmallDecoder). This plugin's
real dependencies are therefore torch/torchvision (already required by Dafne), timm (for
TinyViT's DropPath/to_2tuple/trunc_normal_), and segment-anything-py.

The apply function reimplements evaluate.py from the SegmentAnyMuscle repo slice-by-slice:
  for each slice along the volume's 3rd axis:
    - replicate to 3 channels, resize to the model's fixed 1024x1024 input size
    - min-max normalize, then normalize with ImageNet mean/std
    - run image_encoder -> prompt_encoder (no points/boxes/masks, i.e. automatic/promptless)
      -> mask_decoder with multimask_output=True
    - the mask decoder produces 3 channels (num_classes=3 at construction time); channel
      index 1 is the foreground muscle mask -- this matches evaluate.py's own hardcoded
      `pred_auto[:,1,:]` selection (the meaning of channels 0 and 2 is not documented
      upstream, so this mirrors the only behavior the shipped checkpoint is known to
      produce correctly)
    - resize the logits back to the original slice size and threshold at 0

Unlike evaluate.py, this plugin does not resample by voxel resolution -- matching
evaluate.py, every slice is simply resized to 1024x1024 regardless of physical pixel
spacing, since that is how the model was fine-tuned and evaluated upstream.

There is no command-line configuration for this model (generate_convert() is always run with
no arguments -- see "Run" below); the only per-call configuration point Dafne exposes is the
`'options'` dict passed into apply_model_function via `data.get('options', {})`, declared
under metadata['options']. This model has nothing that needs to be user-configurable, so
`'options'` is unused and left out of the metadata.

License: the SegmentAnyMuscle checkpoint (finetuned_sam.pth) is released under CC BY-NC 4.0
(https://creativecommons.org/licenses/by-nc/4.0/) -- non-commercial use only. This script
itself does not embed or redistribute the weights; it only builds the matching architecture.

Weights: download finetuned_sam.pth from
https://drive.google.com/file/d/1mpTW0TgLgkRIG3sdx9ys5r2iW6lAJw5u/view?usp=sharing
(linked from https://github.com/mazurowski-lab/SegmentAnyMuscle's README) and place it at
weights/weights_segmentanymuscle.pth (relative to this script, i.e. dafne-models/weights/)
before running generate_convert() below -- that is the default_weights_path it loads.

Run:
  python generate_segmentanymuscle_model.py
"""

import os

if 'generate_convert' not in locals() and 'generate_convert' not in globals():
    from dafne_models.common import generate_convert

try:
    from dafne_dl import DynamicTorchModel
except ModuleNotFoundError:
    from dl import DynamicTorchModel


# ---------------------------------------------------------------------------
# init_model_function
# Builds the vit_t (TinyViT) SAM variant with encoder + mask-decoder adapters
# and MoE routing, matching the exact configuration SegmentAnyMuscle's own
# evaluate.py uses to load finetuned_sam.pth. Every class needed to do that is
# defined locally below (ported from SegmentAnyMuscle's sam/modeling/*.py) so
# that this script has no dependency on the SegmentAnyMuscle repo/package.
# ---------------------------------------------------------------------------

def init_segment_any_muscle():
    import itertools
    import math
    import types

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.utils.checkpoint as torch_checkpoint
    from torch.distributions.normal import Normal
    from timm.models.layers import DropPath as TimmDropPath, to_2tuple, trunc_normal_
    from typing import Tuple

    # LayerNorm2d, MLPBlock (common.py), PromptEncoder (prompt_encoder.py, modulo one
    # dead-code line in the box-embedding path this promptless model never exercises),
    # the two-way Attention block, and mask_decoder's hypernetwork MLP are all
    # byte-for-byte (or, for PromptEncoder, behaviorally-for-our-usage) identical to
    # Meta's original segment-anything -- SegmentAnyMuscle never modified them, only
    # added unused sibling classes in the same files. Get them from the real package
    # instead of re-embedding them; only the classes SegmentAnyMuscle actually modified
    # (Adapter/moe_forward, TinyViT, MaskDecoder, TwoWayTransformer/TwoWayAttentionBlock)
    # are defined locally below.
    from segment_anything.modeling.common import LayerNorm2d, MLPBlock
    from segment_anything.modeling.prompt_encoder import PromptEncoder
    from segment_anything.modeling.transformer import Attention
    from segment_anything.modeling.mask_decoder import MLP

    # ---- shared building blocks (sam/modeling/common.py) ----

    class Adapter(nn.Module):
        def __init__(self, D_features, mlp_ratio=0.25, act_layer=nn.GELU, skip_connect=True):
            super().__init__()
            self.skip_connect = skip_connect
            D_hidden_features = int(D_features * mlp_ratio)
            self.act = act_layer()
            self.D_fc1 = nn.Linear(D_features, D_hidden_features)
            self.D_fc2 = nn.Linear(D_hidden_features, D_features)

        def forward(self, x):
            xs = self.D_fc1(x)
            xs = self.act(xs)
            xs = self.D_fc2(xs)
            if self.skip_connect:
                x = x + xs
            else:
                x = xs
            return x

    def moe_forward(self, x, expert, training, noise_epsilon=1e-2):
        B, L, dim = x.shape

        clean_logits = self.gater[-1](x.mean(1))

        if training:
            raw_noise_stddev = self.noise(x.mean(1))
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        logits = logits.softmax(-1)

        try:
            top_logits, top_indices = logits.topk(self.k + 1, dim=1)
        except Exception:
            top_logits, top_indices = logits.topk(self.k, dim=1)
        top_k_logits = top_logits[:, :self.k]
        top_k_indices = top_indices[:, :self.k]
        top_k_gates = top_k_logits / (top_k_logits.sum(1, keepdim=True) + 1e-6)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)

        if training and self.k < top_logits.shape[1]:
            load = self._prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_logits).sum(0)
        else:
            load = self._gates_to_load(gates)

        predicts = []
        for e in expert:
            out = e(x)
            predicts.append(out.unsqueeze(-1))
        predicts = torch.cat(predicts, dim=-1)

        predicts_merged = torch.einsum('blhn,bn->blh', predicts, gates)

        importance = gates.sum(0)
        loss = self.cv_squared(importance) + self.cv_squared(load)
        loss *= 1e-2

        return predicts_merged, loss, gates

    # ---- TinyViT image encoder (sam/modeling/tiny_vit_sam.py) ----

    class Conv2d_BN(torch.nn.Sequential):
        def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1, groups=1, bn_weight_init=1):
            super().__init__()
            self.add_module('c', torch.nn.Conv2d(a, b, ks, stride, pad, dilation, groups, bias=False))
            bn = torch.nn.BatchNorm2d(b)
            torch.nn.init.constant_(bn.weight, bn_weight_init)
            torch.nn.init.constant_(bn.bias, 0)
            self.add_module('bn', bn)

    class DropPath(TimmDropPath):
        def __init__(self, drop_prob=None):
            super().__init__(drop_prob=drop_prob)
            self.drop_prob = drop_prob

    class PatchEmbed(nn.Module):
        def __init__(self, in_chans, embed_dim, resolution, activation):
            super().__init__()
            img_size: Tuple[int, int] = to_2tuple(resolution)
            self.patches_resolution = (img_size[0] // 4, img_size[1] // 4)
            self.num_patches = self.patches_resolution[0] * self.patches_resolution[1]
            self.in_chans = in_chans
            self.embed_dim = embed_dim
            n = embed_dim
            self.seq = nn.Sequential(
                Conv2d_BN(in_chans, n // 2, 3, 2, 1),
                activation(),
                Conv2d_BN(n // 2, n, 3, 2, 1),
            )

        def forward(self, x):
            return self.seq(x)

    class MBConv(nn.Module):
        def __init__(self, in_chans, out_chans, expand_ratio, activation, drop_path):
            super().__init__()
            self.in_chans = in_chans
            self.hidden_chans = int(in_chans * expand_ratio)
            self.out_chans = out_chans

            self.conv1 = Conv2d_BN(in_chans, self.hidden_chans, ks=1)
            self.act1 = activation()
            self.conv2 = Conv2d_BN(self.hidden_chans, self.hidden_chans, ks=3, stride=1, pad=1,
                                    groups=self.hidden_chans)
            self.act2 = activation()
            self.conv3 = Conv2d_BN(self.hidden_chans, out_chans, ks=1, bn_weight_init=0.0)
            self.act3 = activation()
            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        def forward(self, x):
            shortcut = x
            x = self.conv1(x)
            x = self.act1(x)
            x = self.conv2(x)
            x = self.act2(x)
            x = self.conv3(x)
            x = self.drop_path(x)
            x += shortcut
            x = self.act3(x)
            return x

    class PatchMerging(nn.Module):
        def __init__(self, input_resolution, dim, out_dim, activation):
            super().__init__()
            self.input_resolution = input_resolution
            self.dim = dim
            self.out_dim = out_dim
            self.act = activation()
            self.conv1 = Conv2d_BN(dim, out_dim, 1, 1, 0)
            stride_c = 2
            if out_dim in (320, 448, 576):
                stride_c = 1
            self.conv2 = Conv2d_BN(out_dim, out_dim, 3, stride_c, 1, groups=out_dim)
            self.conv3 = Conv2d_BN(out_dim, out_dim, 1, 1, 0)

        def forward(self, x):
            if x.ndim == 3:
                H, W = self.input_resolution
                B = len(x)
                x = x.view(B, H, W, -1).permute(0, 3, 1, 2)
            x = self.conv1(x)
            x = self.act(x)
            x = self.conv2(x)
            x = self.act(x)
            x = self.conv3(x)
            x = x.flatten(2).transpose(1, 2)
            return x

    class ConvLayer(nn.Module):
        def __init__(self, dim, input_resolution, depth, activation, drop_path=0.,
                     downsample=None, use_checkpoint=False, out_dim=None, conv_expand_ratio=4.):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.depth = depth
            self.use_checkpoint = use_checkpoint
            self.blocks = nn.ModuleList([
                MBConv(dim, dim, conv_expand_ratio, activation,
                       drop_path[i] if isinstance(drop_path, list) else drop_path)
                for i in range(depth)
            ])
            if downsample is not None:
                self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
            else:
                self.downsample = None

        def forward(self, x):
            for blk in self.blocks:
                x = torch_checkpoint.checkpoint(blk, x) if self.use_checkpoint else blk(x)
            if self.downsample is not None:
                x = self.downsample(x)
            return x

    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.norm = nn.LayerNorm(in_features)
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.act = act_layer()
            self.drop = nn.Dropout(drop)

        def forward(self, x):
            x = self.norm(x)
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)
            return x

    class TinyViTAttention(torch.nn.Module):
        def __init__(self, dim, key_dim, num_heads=8, attn_ratio=4, resolution=(14, 14)):
            super().__init__()
            assert isinstance(resolution, tuple) and len(resolution) == 2
            self.num_heads = num_heads
            self.scale = key_dim ** -0.5
            self.key_dim = key_dim
            nh_kd = key_dim * num_heads
            self.d = int(attn_ratio * key_dim)
            self.dh = int(attn_ratio * key_dim) * num_heads
            self.attn_ratio = attn_ratio
            h = self.dh + nh_kd * 2

            self.norm = nn.LayerNorm(dim)
            self.qkv = nn.Linear(dim, h)
            self.proj = nn.Linear(self.dh, dim)

            points = list(itertools.product(range(resolution[0]), range(resolution[1])))
            N = len(points)
            attention_offsets = {}
            idxs = []
            for p1 in points:
                for p2 in points:
                    offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                    if offset not in attention_offsets:
                        attention_offsets[offset] = len(attention_offsets)
                    idxs.append(attention_offsets[offset])
            self.attention_biases = torch.nn.Parameter(torch.zeros(num_heads, len(attention_offsets)))
            self.register_buffer('attention_bias_idxs', torch.LongTensor(idxs).view(N, N), persistent=False)

        @torch.no_grad()
        def train(self, mode=True):
            super().train(mode)
            if mode and hasattr(self, 'ab'):
                del self.ab
            else:
                self.register_buffer('ab', self.attention_biases[:, self.attention_bias_idxs], persistent=False)

        def forward(self, x):
            B, N, _ = x.shape
            x = self.norm(x)
            qkv = self.qkv(x)
            q, k, v = qkv.view(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                + (self.attention_biases[:, self.attention_bias_idxs] if self.training else self.ab)
            )
            attn = attn.softmax(dim=-1)
            x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
            x = self.proj(x)
            return x

    class TinyViTBlock(nn.Module):
        def __init__(self, args, dim, input_resolution, num_heads, window_size=7,
                     mlp_ratio=4., drop=0., drop_path=0., depth=1, local_conv_size=3,
                     activation=nn.GELU, moe=-1, k=-1):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.num_heads = num_heads
            assert window_size > 0, 'window_size must be greater than 0'
            self.window_size = window_size
            self.mlp_ratio = mlp_ratio
            self.depth = depth
            self.args = args

            self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

            assert dim % num_heads == 0, 'dim must be divisible by num_heads'
            head_dim = dim // num_heads

            window_resolution = (window_size, window_size)
            self.attn = TinyViTAttention(dim, head_dim, num_heads, attn_ratio=1, resolution=window_resolution)

            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=activation, drop=drop)

            pad = local_conv_size // 2
            self.local_conv = Conv2d_BN(dim, dim, ks=local_conv_size, stride=1, pad=pad, groups=dim)

            if self.args.if_encoder_adapter and (self.depth in self.args.encoder_adapter_depths):
                self.scale = 0.5
                self.moe = moe

                if moe > 0:
                    self.k = k
                    print('Use En-MOE with %s experts and %s selection' % (self.moe, self.k))
                    self.MLP_Adapter = nn.ModuleList([Adapter(dim, skip_connect=False) for _ in range(moe)])
                    self.Space_Adapter = nn.ModuleList([Adapter(dim) for _ in range(moe)])
                    self.gater = nn.ModuleList([nn.Linear(dim, self.moe, bias=False) for _ in range(1)])
                    self.noise = nn.Linear(dim, self.moe, bias=False)
                    self.softplus = nn.Softplus()
                    self.register_buffer("mean", torch.tensor([0.0]))
                    self.register_buffer("std", torch.tensor([1.0]))
                else:
                    self.MLP_Adapter = Adapter(dim, skip_connect=False)
                    self.Space_Adapter = Adapter(dim)

        def _gates_to_load(self, gates):
            return (gates > 0).sum(0)

        def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
            if len(clean_values.shape) == 3:
                clean_values = clean_values.view(-1, clean_values.shape[-1])
                noisy_values = noisy_values.view(-1, noisy_values.shape[-1])
                noise_stddev = noise_stddev.view(-1, noise_stddev.shape[-1])
                noisy_top_values = noisy_top_values.view(-1, noisy_top_values.shape[-1])

            batch = clean_values.size(0)
            m = noisy_top_values.size(1)
            top_values_flat = noisy_top_values.flatten()

            threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
            threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
            is_in = torch.gt(noisy_values, threshold_if_in)
            threshold_positions_if_out = threshold_positions_if_in - 1
            threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

            normal = Normal(self.mean, self.std)
            prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
            prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
            prob = torch.where(is_in, prob_if_in, prob_if_out)
            return prob

        def cv_squared(self, x):
            eps = 1e-10
            return x.float().var() / (x.float().mean() ** 2 + eps)

        def forward(self, x, training=False, moe_print=False):
            moe_total_loss = 0
            gates_total = []

            H, W = self.input_resolution
            B, L, C = x.shape
            assert L == H * W, "input feature has wrong size"

            res_x = x
            if H == self.window_size and W == self.window_size:
                x = self.attn(x)
            else:
                x = x.view(B, H, W, C)
                pad_b = (self.window_size - H % self.window_size) % self.window_size
                pad_r = (self.window_size - W % self.window_size) % self.window_size
                padding = pad_b > 0 or pad_r > 0
                if padding:
                    x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
                pH, pW = H + pad_b, W + pad_r
                nH = pH // self.window_size
                nW = pW // self.window_size
                x = x.view(B, nH, self.window_size, nW, self.window_size, C).transpose(2, 3).reshape(
                    B * nH * nW, self.window_size * self.window_size, C)
                x = self.attn(x)
                x = x.view(B, nH, nW, self.window_size, self.window_size, C).transpose(2, 3).reshape(B, pH, pW, C)
                if padding:
                    x = x[:, :H, :W].contiguous()
                x = x.view(B, L, C)

            if self.args.if_encoder_adapter and (self.depth in self.args.encoder_adapter_depths):
                if self.moe <= 0:
                    x = self.Space_Adapter(x)
                else:
                    moe_out, moe_loss, gates = moe_forward(self, x, self.Space_Adapter, training)
                    moe_total_loss += moe_loss
                    gates_total.append(gates)
                    x = moe_out

            x = res_x + self.drop_path(x)

            x = x.transpose(1, 2).reshape(B, C, H, W)
            x = self.local_conv(x)
            x = x.view(B, C, L).transpose(1, 2)

            if self.args.if_encoder_adapter and (self.depth in self.args.encoder_adapter_depths):
                if self.moe <= 0:
                    x = x + self.drop_path(self.mlp(x)) + self.scale * self.MLP_Adapter(x)
                else:
                    moe_out, moe_loss, gates = moe_forward(self, x, self.MLP_Adapter, training)
                    x = x + self.drop_path(self.mlp(x)) + self.scale * moe_out
                    moe_total_loss += moe_loss
                    gates_total.append(gates)
            else:
                x = x + self.drop_path(self.mlp(x))

            return x, moe_total_loss, gates_total

    class BasicLayer(nn.Module):
        def __init__(self, args, dim, input_resolution, depth, num_heads, window_size,
                     mlp_ratio=4., drop=0., block_idx=0, drop_path=0., downsample=None,
                     use_checkpoint=False, local_conv_size=3, activation=nn.GELU,
                     out_dim=None, moe=-1, k=-1):
            super().__init__()
            self.dim = dim
            self.input_resolution = input_resolution
            self.depth = depth
            self.use_checkpoint = use_checkpoint
            self.args = args

            self.blocks = nn.ModuleList([
                TinyViTBlock(args=self.args, dim=dim, input_resolution=input_resolution,
                             num_heads=num_heads, window_size=window_size, mlp_ratio=mlp_ratio,
                             drop=drop, drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                             depth=block_idx, local_conv_size=local_conv_size, activation=activation,
                             moe=moe, k=k)
                for i in range(depth)
            ])

            if downsample is not None:
                self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, activation=activation)
            else:
                self.downsample = None

        def forward(self, x, training=False, moe_print=False):
            moe_loss = 0
            gates_total = []
            for blk in self.blocks:
                if self.use_checkpoint:
                    x = torch_checkpoint.checkpoint(blk, x)
                else:
                    x, curr_loss, gates = blk(x, training=training, moe_print=moe_print)
                    moe_loss += curr_loss
                    gates_total.append(gates)
            if self.downsample is not None:
                x = self.downsample(x)
            return x, moe_loss, gates_total

    class TinyViT(nn.Module):
        def __init__(self, args, img_size=224, in_chans=3, num_classes=1000,
                     embed_dims=(96, 192, 384, 768), depths=(2, 2, 6, 2),
                     num_heads=(3, 6, 12, 24), window_sizes=(7, 7, 14, 7),
                     mlp_ratio=4., drop_rate=0., drop_path_rate=0.1, use_checkpoint=False,
                     mbconv_expand_ratio=4.0, local_conv_size=3, layer_lr_decay=1.0,
                     moe=-1, k=-1):
            super().__init__()
            self.img_size = img_size
            self.num_classes = num_classes
            self.depths = depths
            self.num_layers = len(depths)
            self.mlp_ratio = mlp_ratio
            self.args = args

            activation = nn.GELU

            self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dims[0],
                                           resolution=img_size, activation=activation)

            patches_resolution = self.patch_embed.patches_resolution
            self.patches_resolution = patches_resolution

            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

            self.layers = nn.ModuleList()
            for i_layer in range(self.num_layers):
                kwargs = dict(
                    dim=embed_dims[i_layer],
                    input_resolution=(
                        patches_resolution[0] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                        patches_resolution[1] // (2 ** (i_layer - 1 if i_layer == 3 else i_layer)),
                    ),
                    depth=depths[i_layer],
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                    use_checkpoint=use_checkpoint,
                    out_dim=embed_dims[min(i_layer + 1, len(embed_dims) - 1)],
                    activation=activation,
                )
                if i_layer == 0:
                    layer = ConvLayer(conv_expand_ratio=mbconv_expand_ratio, **kwargs)
                else:
                    layer = BasicLayer(
                        args=self.args, num_heads=num_heads[i_layer], window_size=window_sizes[i_layer],
                        mlp_ratio=self.mlp_ratio, drop=drop_rate, block_idx=i_layer - 1,
                        local_conv_size=local_conv_size, moe=moe, k=k, **kwargs)
                self.layers.append(layer)

            self.norm_head = nn.LayerNorm(embed_dims[-1])
            self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

            self.apply(self._init_weights)
            self.set_layer_lr_decay(layer_lr_decay)
            self.neck = nn.Sequential(
                nn.Conv2d(embed_dims[-1], 256, kernel_size=1, bias=False),
                LayerNorm2d(256),
                nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                LayerNorm2d(256),
            )

        def set_layer_lr_decay(self, layer_lr_decay):
            decay_rate = layer_lr_decay
            depth = sum(self.depths)
            lr_scales = [decay_rate ** (depth - i - 1) for i in range(depth)]

            def _set_lr_scale(m, scale):
                for p in m.parameters():
                    p.lr_scale = scale

            self.patch_embed.apply(lambda x: _set_lr_scale(x, lr_scales[0]))

            i = 0
            for layer in self.layers:
                for block in layer.blocks:
                    block.apply(lambda x: _set_lr_scale(x, lr_scales[i]))
                    i += 1
                if layer.downsample is not None:
                    layer.downsample.apply(lambda x: _set_lr_scale(x, lr_scales[i - 1]))
            assert i == depth
            for m in [self.norm_head, self.head]:
                m.apply(lambda x: _set_lr_scale(x, lr_scales[-1]))

            for kk, p in self.named_parameters():
                p.param_name = kk

            def _check_lr_scale(m):
                for p in m.parameters():
                    assert hasattr(p, 'lr_scale'), p.param_name

            self.apply(_check_lr_scale)

        def _init_weights(self, m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        def forward_features(self, x, training=False, moe_print=False):
            x = self.patch_embed(x)
            x = self.layers[0](x)
            start_i = 1

            moe_loss_total = 0
            gates_total = []
            for i in range(start_i, len(self.layers)):
                layer = self.layers[i]
                x, moe_loss, gates = layer(x, training=training, moe_print=moe_print)
                moe_loss_total += moe_loss
                gates_total.append(gates)

            B, _, C = x.size()
            x = x.view(B, 64, 64, C)
            x = x.permute(0, 3, 1, 2)
            x = self.neck(x)
            return x, moe_loss_total, gates_total

        def forward(self, x, training=False, moe_print=False):
            x, moe_loss, gates = self.forward_features(x, training=training, moe_print=moe_print)
            return x, moe_loss, gates

    # PositionEmbeddingRandom and PromptEncoder come from segment_anything (imported
    # above) -- SegmentAnyMuscle's own copy is unmodified for the promptless path this
    # plugin uses (points=None, boxes=None, masks=None).

    # ---- mask decoder (sam/modeling/mask_decoder.py) ----
    # MLP (the hypernetwork head) also comes from segment_anything, unmodified.

    class MaskDecoder(nn.Module):
        def __init__(self, *, transformer_dim, transformer, num_multimask_outputs=3,
                     activation=nn.GELU, iou_head_depth=3, iou_head_hidden_dim=256):
            super().__init__()
            self.transformer_dim = transformer_dim
            self.transformer = transformer
            self.num_multimask_outputs = num_multimask_outputs

            self.iou_token = nn.Embedding(1, transformer_dim)
            self.num_mask_tokens = num_multimask_outputs + 1
            self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

            self.output_upscaling = nn.Sequential(
                nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
                LayerNorm2d(transformer_dim // 4),
                activation(),
                nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
                activation(),
            )
            self.output_hypernetworks_mlps = nn.ModuleList([
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ])
            self.iou_prediction_head = MLP(transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth)

        def forward(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                    dense_prompt_embeddings, multimask_output, training=False):
            masks, iou_pred, moe_loss, gates = self.predict_masks(
                image_embeddings=image_embeddings,
                image_pe=image_pe,
                sparse_prompt_embeddings=sparse_prompt_embeddings,
                dense_prompt_embeddings=dense_prompt_embeddings,
                training=training,
            )

            mask_slice = slice(1, None) if multimask_output else slice(0, 1)
            masks = masks[:, mask_slice, :, :]
            iou_pred = iou_pred[:, mask_slice]

            return masks, iou_pred, moe_loss, gates

        def predict_masks(self, image_embeddings, image_pe, sparse_prompt_embeddings,
                           dense_prompt_embeddings, training=False):
            output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
            output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)
            tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

            if image_embeddings.shape[0] != tokens.shape[0]:
                src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
            else:
                src = image_embeddings
            src = src + dense_prompt_embeddings
            pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
            b, c, h, w = src.shape

            hs, src, moe_loss, gates = self.transformer(src, pos_src, tokens, training=training)
            iou_token_out = hs[:, 0, :]
            mask_tokens_out = hs[:, 1:(1 + self.num_mask_tokens), :]

            src = src.transpose(1, 2).view(b, c, h, w)
            upscaled_embedding = self.output_upscaling(src)
            hyper_in_list = []
            for i in range(self.num_mask_tokens):
                hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
            hyper_in = torch.stack(hyper_in_list, dim=1)
            b, c, h, w = upscaled_embedding.shape
            masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

            iou_pred = self.iou_prediction_head(iou_token_out)

            return masks, iou_pred, moe_loss, gates

    # ---- two-way transformer (sam/modeling/transformer.py) ----
    # Attention (used below as self_attn/cross_attn_token_to_image/
    # cross_attn_image_to_token/final_attn_token_to_image) also comes from
    # segment_anything, unmodified.

    class TwoWayAttentionBlock(nn.Module):
        def __init__(self, args, embedding_dim, num_heads, mlp_dim=2048, activation=nn.ReLU,
                     attention_downsample_rate=2, if_adapter=False, skip_first_layer_pe=False,
                     moe=-1, k=-1, depth=0):
            super().__init__()
            self.args = args
            self.if_adapter = if_adapter
            self.self_attn = Attention(embedding_dim, num_heads)
            self.norm1 = nn.LayerNorm(embedding_dim)

            self.cross_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
            self.norm2 = nn.LayerNorm(embedding_dim)

            self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
            self.norm3 = nn.LayerNorm(embedding_dim)

            self.norm4 = nn.LayerNorm(embedding_dim)
            self.cross_attn_image_to_token = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)

            if self.if_adapter:
                self.scale = 0.5
                self.moe = moe

                if moe > 0:
                    self.k = k
                    print('Use Dn-MOE with %s experts and %s selection' % (self.moe, self.k))
                    self.MLP_Adapter = nn.ModuleList([Adapter(embedding_dim, skip_connect=False) for _ in range(moe)])
                    self.Adapter = nn.ModuleList([Adapter(embedding_dim) for _ in range(moe)])
                    self.Adapter2 = nn.ModuleList([Adapter(embedding_dim) for _ in range(moe)])
                    self.gater = nn.ModuleList([nn.Linear(embedding_dim, self.moe, bias=False) for _ in range(1)])
                    self.noise = nn.Linear(embedding_dim, self.moe, bias=False)
                    self.softplus = nn.Softplus()
                    self.register_buffer("mean", torch.tensor([0.0]))
                    self.register_buffer("std", torch.tensor([1.0]))
                else:
                    self.MLP_Adapter = Adapter(embedding_dim, skip_connect=False)
                    self.Adapter = Adapter(embedding_dim)
                    self.Adapter2 = Adapter(embedding_dim)

            self.skip_first_layer_pe = skip_first_layer_pe

        def _gates_to_load(self, gates):
            return (gates > 0).sum(0)

        def _prob_in_top_k(self, clean_values, noisy_values, noise_stddev, noisy_top_values):
            batch = clean_values.size(0)
            m = noisy_top_values.size(1)
            top_values_flat = noisy_top_values.flatten()

            threshold_positions_if_in = torch.arange(batch, device=clean_values.device) * m + self.k
            threshold_if_in = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_in), 1)
            is_in = torch.gt(noisy_values, threshold_if_in)
            threshold_positions_if_out = threshold_positions_if_in - 1
            threshold_if_out = torch.unsqueeze(torch.gather(top_values_flat, 0, threshold_positions_if_out), 1)

            normal = Normal(self.mean, self.std)
            prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
            prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
            prob = torch.where(is_in, prob_if_in, prob_if_out)
            return prob

        def cv_squared(self, x):
            eps = 1e-10
            return x.float().var() / (x.float().mean() ** 2 + eps)

        def forward(self, queries, keys, query_pe, key_pe, labels=None, training=False):
            moe_loss_total = 0
            gates_total = []

            if self.skip_first_layer_pe:
                queries = self.self_attn(q=queries, k=queries, v=queries)
            else:
                q = queries + query_pe
                attn_out = self.self_attn(q=q, k=q, v=queries)
                queries = queries + attn_out
            queries = self.norm1(queries)

            q = queries + query_pe
            k = keys + key_pe
            attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
            queries = queries + attn_out

            if self.if_adapter:
                if self.moe <= 0:
                    queries = self.Adapter(queries)
                else:
                    moe_out, moe_loss, gates = moe_forward(self, queries, self.Adapter, training)
                    queries = moe_out
                    moe_loss_total += moe_loss
                    gates_total.append(gates)

            queries = self.norm2(queries)

            mlp_out = self.mlp(queries)
            if self.if_adapter:
                if self.moe <= 0:
                    queries = queries + mlp_out + self.scale * self.MLP_Adapter(queries)
                else:
                    moe_out, moe_loss, gates = moe_forward(self, queries, self.MLP_Adapter, training)
                    queries = queries + mlp_out + self.scale * moe_out
                    moe_loss_total += moe_loss
                    gates_total.append(gates)
            else:
                queries = queries + mlp_out
            queries = self.norm3(queries)

            q = queries + query_pe
            k = keys + key_pe
            attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
            keys = keys + attn_out

            if self.if_adapter:
                if self.moe <= 0:
                    keys = self.Adapter2(keys)
                else:
                    moe_out, moe_loss, gates = moe_forward(self, keys, self.Adapter2, training)
                    moe_loss_total += moe_loss
                    gates_total.append(gates)
                    keys = moe_out

            keys = self.norm4(keys)

            return queries, keys, moe_loss_total, gates_total

    class TwoWayTransformer(nn.Module):
        def __init__(self, args, depth, embedding_dim, num_heads, mlp_dim,
                     activation=nn.ReLU, attention_downsample_rate=2, moe=-1, k=-1):
            super().__init__()
            self.args = args
            self.depth = depth
            self.embedding_dim = embedding_dim
            self.num_heads = num_heads
            self.mlp_dim = mlp_dim
            self.layers = nn.ModuleList()
            for i in range(depth):
                if_adapter = args.if_mask_decoder_adapter if i < args.decoder_adapt_depth else False
                self.layers.append(
                    TwoWayAttentionBlock(
                        args=self.args, embedding_dim=embedding_dim, num_heads=num_heads,
                        mlp_dim=mlp_dim, activation=activation,
                        attention_downsample_rate=attention_downsample_rate,
                        if_adapter=if_adapter, skip_first_layer_pe=(i == 0),
                        moe=moe, k=k, depth=i,
                    )
                )

            self.final_attn_token_to_image = Attention(embedding_dim, num_heads, downsample_rate=attention_downsample_rate)
            self.norm_final_attn = nn.LayerNorm(embedding_dim)

        def forward(self, image_embedding, image_pe, point_embedding, training=False):
            bs, c, h, w = image_embedding.shape
            image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
            image_pe = image_pe.flatten(2).permute(0, 2, 1)

            queries = point_embedding
            keys = image_embedding

            moe_total_loss = 0
            gates_total = []
            for layer in self.layers:
                queries, keys, moe_loss, gates = layer(
                    queries=queries, keys=keys, query_pe=point_embedding, key_pe=image_pe, training=training,
                )
                moe_total_loss += moe_loss
                gates_total.append(gates)

            q = queries + point_embedding
            k = keys + image_pe
            attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys)
            queries = queries + attn_out
            queries = self.norm_final_attn(queries)

            return queries, keys, moe_total_loss, gates_total

    # ---- container tying the three sub-modules together with the attribute
    # names ("image_encoder", "prompt_encoder", "mask_decoder") that the
    # checkpoint's state_dict keys are prefixed with ----

    class SegmentAnyMuscleNet(nn.Module):
        def __init__(self, image_encoder, prompt_encoder, mask_decoder):
            super().__init__()
            self.image_encoder = image_encoder
            self.prompt_encoder = prompt_encoder
            self.mask_decoder = mask_decoder

    # ---- assemble the vit_t (TinyViT) variant, matching build_sam_vit_t() /
    # evaluate.py's configuration for finetuned_sam.pth ----

    args = types.SimpleNamespace(
        if_encoder_adapter=True,
        encoder_adapter_depths=[0, 1, 10, 11],
        if_mask_decoder_adapter=True,
        decoder_adapt_depth=2,
    )
    moe, k, num_classes = 10, 10, 3
    prompt_embed_dim = 256
    image_size = 1024
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size

    image_encoder = TinyViT(
        args, img_size=image_size, in_chans=3, num_classes=1000,
        embed_dims=[64, 128, 160, 320],
        depths=[2, 2, 6, 2],
        num_heads=[2, 4, 5, 10],
        window_sizes=[7, 7, 14, 7],
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.0,
        use_checkpoint=False,
        mbconv_expand_ratio=4.0,
        local_conv_size=3,
        layer_lr_decay=0.8,
        moe=moe,
        k=k,
    )
    prompt_encoder = PromptEncoder(
        embed_dim=prompt_embed_dim,
        image_embedding_size=(image_embedding_size, image_embedding_size),
        input_image_size=(image_size, image_size),
        mask_in_chans=16,
    )
    mask_decoder = MaskDecoder(
        num_multimask_outputs=num_classes,
        transformer=TwoWayTransformer(
            args=args, depth=2, embedding_dim=prompt_embed_dim, mlp_dim=2048, num_heads=8, moe=moe, k=k,
        ),
        transformer_dim=prompt_embed_dim,
        iou_head_depth=3,
        iou_head_hidden_dim=256,
    )

    return SegmentAnyMuscleNet(image_encoder, prompt_encoder, mask_decoder)


# ---------------------------------------------------------------------------
# apply_model_function
# All imports and constants must be local -- this function's source is
# serialized (via dill/inspect.getsource) and re-executed standalone.
# ---------------------------------------------------------------------------

def segment_any_muscle_apply(modelObj, data: dict):
    import numpy as np
    import torch
    from torchvision import transforms
    from dafne_dl.interfaces import WrongDimensionalityError

    MODEL_IMAGE_SIZE = 1024
    MASK_CHANNEL = 1  # foreground muscle channel; see module docstring
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    image = np.asarray(data['image'], dtype=np.float32)  # (H, W, D)
    if image.ndim != 3:
        raise WrongDimensionalityError('SegmentAnyMuscle expects a 3D volume')

    h, w, n_slices = image.shape

    model = modelObj.model
    model.eval()
    device = modelObj.device

    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    resize_to_model = transforms.Resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE))
    resize_to_native = transforms.Resize((h, w))

    mask_vol = np.zeros((h, w, n_slices), dtype=np.uint8)

    with torch.no_grad():
        for idx in range(n_slices):
            slice_2d = torch.as_tensor(image[:, :, idx], dtype=torch.float32)
            slice_3c = slice_2d.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
            slice_3c = resize_to_model(slice_3c)
            slice_3c = (slice_3c - slice_3c.min()) / (slice_3c.max() - slice_3c.min() + 1e-8)
            slice_3c = normalize(slice_3c)
            input_tensor = slice_3c.unsqueeze(0).to(device)  # (1, 3, 1024, 1024)

            img_emb, _, _ = model.image_encoder(input_tensor, None)
            sparse_emb, dense_emb = model.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
            pred, _, _, _ = model.mask_decoder(
                image_embeddings=img_emb,
                image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_emb,
                dense_prompt_embeddings=dense_emb,
                multimask_output=True,
            )
            pred = pred[:, MASK_CHANNEL, :, :]  # (1, 256, 256)
            pred = resize_to_native(pred)
            mask_vol[:, :, idx] = (pred[0] >= 0).cpu().numpy().astype(np.uint8)

    return {'Muscle': mask_vol}


metadata = {
    'categories': [['MSK', 'Muscle']],
    'variants': [''],
    'dimensionality': '3',
    'model_name': 'SegmentAnyMuscle',
    'model_type': 'DynamicTorchModel',
    'orientation': '',
    'info': {
        'Description': 'Automatic (promptless) muscle segmentation in MRI/CT volumes using a '
                        'fine-tuned Segment Anything Model (MobileSAM/TinyViT backbone with '
                        'encoder and mask-decoder adapter layers, plus mixture-of-experts '
                        'routing), applied slice-by-slice over the volume.\n'
                        'Note: Not for commercial use!',
        'Author': 'Colglazier, Lee, Dong et al.',
        'Modality': 'MRI',
        'Reference': 'https://github.com/mazurowski-lab/SegmentAnyMuscle',
        'License': 'CC BY-NC 4.0',
    },
    'dependencies': {
        'timm': 'timm >= 0.9 --triton',
        'segment_anything': 'segment-anything-py --triton',
    },
}

generate_convert(
    model_id='ad221302-115e-41f5-a7bd-711d315ac6aa',
    default_weights_path=os.path.join('weights', 'weights_segmentanymuscle.pth'),
    model_name_prefix='SegmentAnyMuscle',
    model_create_function=init_segment_any_muscle,
    model_apply_function=segment_any_muscle_apply,
    model_learn_function=None,
    dimensionality=3,
    model_type=DynamicTorchModel,
    metadata=metadata,
)
