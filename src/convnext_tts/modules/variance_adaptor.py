import torch
import torch.nn as nn
from convnext_tts.layers.alignment import AlignmentModule, viterbi_decode
from convnext_tts.layers.convnext import ConvNeXtLayer
from convnext_tts.losses.forwardsum import ForwardSumLoss
from convnext_tts.utils.model import generate_path, sequence_mask


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

    def forward(self, x, mask):
        if self.detach:
            x = x.detach()
        for layer in self.layers:
            x = layer(x) * mask
        o = self.out_conv(x) * mask
        return o


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

    def forward(self, x, y, log_cf0, x_mask, y_mask, x_lengths, y_lengths):
        # P: phoneme level, T: frame level
        # x: (B, C, P)
        # y: (B, C, T)
        # log_cf0: (B, 1, T)
        # x_mask: (B, 1, P) phoneme level mask
        # y_mask: (B, 1, T) frame level mask
        # x_lengths: (B,)
        # y_lengths: (B,)

        log_duration_pred = self.duration_predictor(x, x_mask)

        log_p_attn = self.alignment_module(
            text=x.transpose(1, 2),
            feats=y.transpose(1, 2),
            x_masks=x_mask.squeeze(1).bool().logical_not(),
        )
        duration, loss_bin = viterbi_decode(log_p_attn, x_lengths, y_lengths)
        loss_forwardsum = self.forwardsum_loss(log_p_attn, x_lengths, y_lengths)
        duration = duration.unsqueeze(1)
        path_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))
        x_frame = x @ attn_path
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
        )

    def infer(self, x, x_mask):
        log_duration = self.duration_predictor(x)
        duration = log_duration.exp().round().long()

        y_lengths = duration.sum(dim=[1, 2])
        y_mask = sequence_mask(y_lengths).unsqueeze(1).to(x.device)
        path_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))

        x_frame = attn_path @ x
        log_cf0_vuv = self.pitch_predictor(x_frame)
        log_cf0, vuv = log_cf0_vuv.split(1, dim=1)
        vuv = vuv.sigmoid()
        vuv = torch.where(vuv < 0.5, 0.0, 1.0)
        pitch_emb = self.pitch_emb(torch.zeros_like(log_cf0)) * y_mask
        x_frame = x_frame + pitch_emb
        return x_frame, (duration, log_cf0, vuv)
