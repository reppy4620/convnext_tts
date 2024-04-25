import hydra
import torch
from hydra.utils import instantiate
from lightning import Trainer


def test_model(cfg):
    model = instantiate(cfg.generator)
    hop_length = cfg.mel.hop_length
    n_mels = cfg.mel.n_mels

    B = 4
    P = 10
    F = 100
    S = F * hop_length

    bnames = ["a"] * B
    phoneme = torch.randint(1, 30, (B, P))
    mel = torch.randn(B, n_mels, F)
    cf0 = torch.randn(B, 1, F).abs()
    vuv = torch.ones(B, 1, F).float()
    wav = torch.rand(B, 1, S)
    phone_lengths = torch.tensor([P] * 4)
    frame_lengths = torch.tensor([F] * 4)
    sample_lengths = torch.tensor([S] * 4)

    batch = (
        bnames,
        phoneme,
        mel,
        cf0,
        vuv,
        wav,
        phone_lengths,
        frame_lengths,
        sample_lengths,
    )

    (
        wav_pred,
        (loss_duration, loss_log_cf0, loss_vuv, loss_forwardsum, loss_bin),
        idx_start,
    ) = model.training_step(batch)
    print(wav_pred.shape)
    print(loss_duration, loss_log_cf0, loss_vuv, loss_forwardsum, loss_bin)
    print(idx_start)


def test_train(cfg):
    lit_module = instantiate(cfg.lit_module, params=cfg, _recursive_=False)
    train_dl = lit_module.train_dataloader()
    print(next(iter(train_dl)))
    trainer = Trainer(
        max_steps=1,
        detect_anomaly=True,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model=lit_module)


@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg):
    print(cfg)
    test_model(cfg)
    # test_train(cfg)


if __name__ == "__main__":
    main()
