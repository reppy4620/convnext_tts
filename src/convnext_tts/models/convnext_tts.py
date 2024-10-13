import torch.nn as nn

from convnext_tts.layers.embedding import EmbeddingLayer
from convnext_tts.losses.masked import masked_l1_loss, masked_mse_loss
from convnext_tts.modules.convnext import ConvNeXtModule
from convnext_tts.modules.variance_adaptor import VarianceAdaptor
from convnext_tts.modules.vocoder import Vocoder
from convnext_tts.utils.model import rand_slice_segments, sequence_mask, to_log_scale


class ConvNeXtTTS(nn.Module):
    def __init__(
        self,
        embedding: EmbeddingLayer,
        encoder: ConvNeXtModule,
        variance_adaptor: VarianceAdaptor,
        decoder: ConvNeXtModule,
        vocoder: Vocoder,
        frame_segment_size: int,
    ):
        super().__init__()
        self.frame_segment_size = frame_segment_size

        self.embedding = embedding
        self.encoder = encoder
        self.variance_adaptor = variance_adaptor
        self.decoder = decoder
        self.vocoder = vocoder

    def training_step(self, batch):
        _, phoneme, mel, cf0, vuv, _, phone_lengths, frame_lengths, _ = batch
        phone_mask = sequence_mask(phone_lengths).unsqueeze(1).to(mel.dtype)
        frame_mask = sequence_mask(frame_lengths).unsqueeze(1).to(mel.dtype)
        log_cf0 = to_log_scale(cf0)

        x = self.embedding(phoneme, phone_mask)
        x = self.encoder(x, phone_mask)
        (
            x_frame,
            duration,
            (log_duration_pred, log_cf0_pred, vuv_pred),
            (loss_forwardsum, loss_bin),
            p_attn,
        ) = self.variance_adaptor(
            x=x,
            y=mel,
            log_cf0=log_cf0,
            x_mask=phone_mask,
            y_mask=frame_mask,
            x_lengths=phone_lengths,
            y_lengths=frame_lengths,
        )
        x_frame = self.decoder(x_frame, frame_mask)
        x_frame_crop, idx_start = rand_slice_segments(
            x_frame, lengths=frame_lengths, segment_size=self.frame_segment_size
        )
        wav_pred = self.vocoder(x_frame_crop)

        log_duration = to_log_scale(duration)
        loss_duration = masked_mse_loss(log_duration_pred, log_duration, phone_mask)

        loss_log_cf0 = masked_l1_loss(log_cf0_pred, log_cf0, frame_mask)
        loss_vuv = masked_l1_loss(vuv_pred, vuv, frame_mask)
        return (
            wav_pred,
            (loss_duration, loss_log_cf0, loss_vuv, loss_forwardsum, loss_bin),
            idx_start,
            p_attn,
        )

    def forward(self, phoneme):
        # phoneme: (B, P)
        phone_lengths = (phoneme != 0).sum(dim=1)
        phone_mask = sequence_mask(phone_lengths)
        x = self.embedding(phoneme, phone_mask)
        x = self.encoder(x, phone_mask)
        x_frame, frame_mask, (duration, log_cf0, vuv) = self.variance_adaptor.infer(
            x=x, x_mask=phone_mask
        )
        x_frame = self.decoder(x_frame, frame_mask)
        wav = self.vocoder(x_frame)
        return wav.squeeze(1), (duration, log_cf0, vuv)
