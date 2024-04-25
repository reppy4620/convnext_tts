import torch.nn.functional as F
from convnext_tts.losses.gan import discriminator_loss, feature_loss, generator_loss
from convnext_tts.utils.dataset import ShuffleBatchSampler, batch_by_size
from convnext_tts.utils.logging import logger
from convnext_tts.utils.model import slice_segments
from hydra.utils import instantiate
from lightning import LightningModule
from torch.utils.data import DataLoader


class NormalLitModule(LightningModule):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.loss_coef = params.train.loss_coef

        self.frame_segment_size = params.train.frame_segment_size
        self.sample_segment_size = params.train.sample_segment_size

        self.automatic_optimization = False

        self.generator = instantiate(params.generator)
        self.discriminator = instantiate(params.discriminator)
        logger.info(
            f"Generator: {sum(p.numel() for p in self.generator.parameters()) / 1e6:.3f}M"
        )
        logger.info(
            f"Discriminator: {sum(p.numel() for p in self.discriminator.parameters()) / 1e6:.3f}M"
        )

        self.to_mel = instantiate(params.mel)
        self.hop_length = params.mel.hop_length

        self.collator = instantiate(params.dataset.collator)

    def forward(self, phoneme):
        return self.generator(phoneme).squeeze(1)

    def _handle_batch(self, batch, train=True):
        optimizer_g, optimizer_d = self.optimizers()

        (
            wav_hat,
            (loss_duration, loss_cf0, loss_vuv, loss_forwardsum, loss_bin),
            idx_start,
        ) = self.generator.training_step(batch)
        mel_hat = self.to_mel(wav_hat.squeeze(1))

        mel = batch[2]
        mel = slice_segments(
            mel, start_indices=idx_start, segment_size=self.frame_segment_size
        )
        wav = batch[5]
        wav = slice_segments(
            wav,
            start_indices=idx_start * self.hop_length,
            segment_size=self.sample_segment_size,
        )
        d_real, d_fake, _, _ = self.discriminator(wav, wav_hat.detach())
        loss_d = discriminator_loss(d_real, d_fake)
        if train:
            optimizer_d.zero_grad()
            self.manual_backward(loss_d)
            optimizer_d.step()

        _, d_fake, fmap_real, fmap_fake = self.discriminator(wav, wav_hat)
        loss_gen = generator_loss(d_fake)
        loss_mel = self.loss_coef.mel * F.l1_loss(mel_hat, mel)
        loss_fm = self.loss_coef.fm * feature_loss(fmap_real, fmap_fake)
        loss_gan = loss_gen + loss_mel + loss_fm
        loss_var = loss_duration + loss_cf0 + loss_vuv
        loss_align = loss_forwardsum + loss_bin
        loss_g = (
            loss_gan + self.loss_coef.var * loss_var + self.loss_coef.align * loss_align
        )
        if train:
            optimizer_g.zero_grad()
            self.manual_backward(loss_g)
            optimizer_g.step()

        loss_dict = dict(
            disc=loss_d,
            gen=loss_gen,
            mel=loss_mel,
            fm=loss_fm,
            dur=loss_duration,
            cf0=loss_cf0,
            vuv=loss_vuv,
            fowardsum=loss_forwardsum,
            bin=loss_bin,
        )

        self.log_dict(loss_dict, prog_bar=True)

    def training_step(self, batch):
        self._handle_batch(batch, train=True)

    def on_train_epoch_end(self):
        scheduler_g, scheduler_d = self.lr_schedulers()
        scheduler_g.step()
        scheduler_d.step()

    def validation_step(self, batch, batch_idx):
        self._handle_batch(batch, train=False)

    def train_dataloader(self):
        train_ds = instantiate(self.params.dataset.train)
        indices = train_ds.ordered_indices()
        batches = batch_by_size(
            indices=indices,
            num_tokens_fn=train_ds.num_tokens,
            max_tokens=self.params.dataset.max_tokens,
            required_batch_size_multiple=1,
        )
        batch_sampler = ShuffleBatchSampler(batches, drop_last=True, shuffle=True)
        train_dl = DataLoader(
            train_ds,
            num_workers=self.params.train.num_workers,
            pin_memory=True,
            prefetch_factor=2,
            collate_fn=self.collator,
            batch_sampler=batch_sampler,
        )
        return train_dl

    def val_dataloader(self):
        val_ds = instantiate(self.params.dataset.valid)
        val_dl = DataLoader(
            val_ds,
            batch_size=self.params.train.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.params.train.num_workers,
            pin_memory=True,
            collate_fn=self.collator,
        )
        return val_dl

    def configure_optimizers(self):
        optimizer_g = instantiate(
            self.params.optimizer, params=self.generator.parameters()
        )
        optimizer_d = instantiate(
            self.params.optimizer, params=self.discriminator.parameters()
        )
        scheduler_g = instantiate(self.params.scheduler, optimizer=optimizer_g)
        scheduler_d = instantiate(self.params.scheduler, optimizer=optimizer_d)
        return [optimizer_g, optimizer_d], [scheduler_g, scheduler_d]
