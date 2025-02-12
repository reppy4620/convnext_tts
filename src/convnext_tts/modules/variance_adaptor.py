from typing import Tuple

import torch
import torch.nn as nn

from convnext_tts.layers.alignment import AlignmentModule, viterbi_decode
from convnext_tts.layers.convnext import ConvNeXtLayer
from convnext_tts.losses.forwardsum import ForwardSumLoss
from convnext_tts.utils.model import length_to_mask
from convnext_tts.utils.typing import Float, Int


# adapted from https://github.com/espnet/espnet/blob/master/espnet2/gan_tts/jets/length_regulator.py
# Copyright 2022 Dan Lim
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
class GaussianUpsampling(torch.nn.Module):
    """Gaussian upsampling with fixed temperature as in:

    https://arxiv.org/abs/2010.04301

    """

    def __init__(self, delta=0.1):
        super().__init__()
        self.delta = delta

    def forward(self, hs, ds, h_masks=None, d_masks=None):
        """Upsample hidden states according to durations.

        Args:
            hs (Tensor): Batched hidden state to be expanded (B, T_text, adim).
            ds (Tensor): Batched token duration (B, T_text).
            h_masks (Tensor): Mask tensor (B, T_feats).
            d_masks (Tensor): Mask tensor (B, T_text).

        Returns:
            Tensor: Expanded hidden state (B, T_feat, adim).

        """
        B = ds.size(0)
        device = ds.device

        if ds.sum() == 0:
            # NOTE(kan-bayashi): This case must not be happened in teacher forcing.
            #   It will be happened in inference with a bad duration predictor.
            #   So we do not need to care the padded sequence case here.
            ds[ds.sum(dim=1).eq(0)] = 1

        if h_masks is None:
            T_feats = ds.sum().int()
        else:
            T_feats = h_masks.size(-1)
        t = torch.arange(0, T_feats).unsqueeze(0).repeat(B, 1).to(device).float()
        if h_masks is not None:
            t = t * h_masks.float()

        c = ds.cumsum(dim=-1) - ds / 2
        energy = -1 * self.delta * (t.unsqueeze(-1) - c.unsqueeze(1)) ** 2
        if d_masks is not None:
            energy = energy.masked_fill(
                ~(d_masks.unsqueeze(1).repeat(1, T_feats, 1)), -float("inf")
            )

        p_attn = torch.softmax(energy, dim=2)  # (B, T_feats, T_text)
        hs = torch.matmul(p_attn, hs)
        return hs, p_attn


class VariancePredictor(nn.Module):
    def __init__(self, channels, out_channels, num_layers, detach=False):
        super().__init__()
        self.detach = detach

        self.layers = nn.ModuleList(
            [
                ConvNeXtLayer(channels, channels, 1 / num_layers)
                for _ in range(num_layers)
            ]
        )
        self.out_conv = nn.Conv1d(channels, out_channels, 1)

    def forward(self, x: Float["b c t"], mask: Float["b 1 t"]) -> Float["b c t"]:
        if self.detach:
            x = x.detach()
        for layer in self.layers:
            x = layer(x) * mask
        o = self.out_conv(x) * mask
        return o


# P: phoneme length, T: Frame length
VarianceAdaptorOutput = Tuple[
    Float["b c t"],  # x_frame
    Float["b 1 p"],  # duration
    Tuple[
        Float["b 1 p"], Float["b 1 t"], Float["b 1 t"]
    ],  # (log_duration_pred, log_cf0_pred, vuv_pred)
    Tuple[Float, Float],  # (loss_bin, loss_forwardsum)
    Float["b t p"],  # p_attn
]


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        duration_predictor: VariancePredictor,
        alignment_module: AlignmentModule,
        pitch_predictor: VariancePredictor,
        pitch_emb: nn.Conv1d,
        forwardsum_loss: ForwardSumLoss,
    ):
        super().__init__()
        self.duration_predictor = duration_predictor
        self.alignment_module = alignment_module
        self.pitch_predictor = pitch_predictor
        self.pitch_emb = pitch_emb
        self.forwardsum_loss = forwardsum_loss

        self.length_regulator = GaussianUpsampling()

    def forward(
        self,
        x: Float["B C P"],
        y: Float["B C T"],
        log_cf0: Float["B 1 T"],
        x_mask: Float["B 1 P"],
        y_mask: Float["B 1 P"],
        x_lengths: Int["B"],
        y_lengths: Int["B"],
    ) -> VarianceAdaptorOutput:
        log_duration_pred = self.duration_predictor(x, x_mask)

        log_p_attn = self.alignment_module(
            text=x.transpose(1, 2),
            feats=y.transpose(1, 2),
            text_lengths=x_lengths,
            feats_lengths=y_lengths,
            x_masks=x_mask.squeeze(1).bool().logical_not(),
        )
        duration, loss_bin = viterbi_decode(log_p_attn, x_lengths, y_lengths)
        loss_forwardsum = self.forwardsum_loss(log_p_attn, x_lengths, y_lengths)
        x_frame, p_attn = self.length_regulator(
            hs=x.transpose(1, 2),
            ds=duration,
            h_masks=y_mask.squeeze(1).bool(),
            d_masks=x_mask.squeeze(1).bool(),
        )
        # [B, T, C] -> [B, C, T]
        x_frame = x_frame.transpose(1, 2)

        assert x_frame.shape[-1] == y.shape[-1], f"{x_frame.shape} != {y.shape}"

        log_cf0_vuv_pred = self.pitch_predictor(x_frame, y_mask)
        log_cf0_pred, vuv_pred = log_cf0_vuv_pred.split(1, dim=1)
        vuv_pred = vuv_pred.sigmoid()
        pitch_emb = self.pitch_emb(log_cf0) * y_mask
        x_frame = x_frame + pitch_emb
        return (
            x_frame,
            duration,
            (log_duration_pred, log_cf0_pred, vuv_pred),
            (loss_bin, loss_forwardsum),
            p_attn,
        )

    def infer(
        self, x: Float["B C P"], x_mask: Float["B 1 P"]
    ) -> Tuple[
        Float["B C T"],
        Float["B 1 T"],
        Tuple[Float["B 1 P"], Float["B 1 T"], Float["B 1 T"]],
    ]:
        log_duration = self.duration_predictor(x, x_mask)
        duration = log_duration.exp().round().long()
        y_lengths = duration.sum(dim=[1, 2])
        y_mask = length_to_mask(y_lengths).unsqueeze(1).to(x.dtype)

        x_frame, p_attn = self.length_regulator(
            hs=x.transpose(1, 2),
            ds=duration.squeeze(1),
            h_masks=y_mask.squeeze(1).bool(),
            d_masks=x_mask.squeeze(1).bool(),
        )
        x_frame = x_frame.transpose(1, 2)
        log_cf0_vuv = self.pitch_predictor(x_frame, y_mask)
        log_cf0, vuv = log_cf0_vuv.split(1, dim=1)
        vuv = vuv.sigmoid()
        vuv = torch.where(vuv < 0.5, 0.0, 1.0)
        pitch_emb = self.pitch_emb(log_cf0) * y_mask
        x_frame = x_frame + pitch_emb
        return x_frame, y_mask, (duration, log_cf0, vuv)
