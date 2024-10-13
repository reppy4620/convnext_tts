import time
from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
import torchaudio
from hydra.utils import get_class, instantiate
from tqdm import tqdm

from convnext_tts.utils.logging import logger


@hydra.main(config_path="conf", version_base=None, config_name="config")
@torch.inference_mode()
def main(cfg):
    out_dir = Path(cfg.syn.out_dir)
    [(out_dir / s).mkdir(parents=True, exist_ok=True) for s in ["wav"]]
    wav_dir = Path(cfg.path.wav_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading checkpoint... : {cfg.syn.ckpt_path}")
    lit_module = (
        get_class(cfg.lit_module._target_)
        .load_from_checkpoint(cfg.syn.ckpt_path, params=cfg)
        .to(device)
        .eval()
    )

    mos_predictor = (
        torch.hub.load("tarepan/SpeechMOS:v1.0.0", "utmos22_strong", trust_repo=True)
        .to(device)
        .eval()
    )

    bnames = []
    rtf_list = []
    mos_list = []

    test_ds = instantiate(cfg.dataset.valid)

    logger.info(f"Start synthesis... : {len(test_ds)} samples")
    for inputs in tqdm(test_ds, total=len(test_ds)):
        bname, *_ = inputs
        bnames.append(bname)
        torch.cuda.synchronize()
        s = time.time()
        o, _ = lit_module(inputs[1:])
        torch.cuda.synchronize()
        rtf = (time.time() - s) / o.shape[-1] * cfg.mel.sample_rate
        rtf_list.append(rtf)
        torchaudio.save(
            out_dir / f"wav/{bname}.wav",
            o.cpu(),
            cfg.mel.sample_rate,
        )

        wav_path = wav_dir / f"{bname}.wav"
        wav, sr = torchaudio.load(wav_path)
        assert sr == cfg.mel.sample_rate
        wav = wav.to(device)
        # calculate UTMOS
        gt_mos = mos_predictor(wav, sr)
        syn_mos = mos_predictor(o, sr)
        mos_list.append((gt_mos.item(), syn_mos.item()))

    df = pd.DataFrame(
        {
            "fname": bnames,
            "rtf": rtf_list,
            "gt_mos": [x[0] for x in mos_list],
            "syn_mos": [x[1] for x in mos_list],
        }
    )
    df.to_csv(out_dir / "result.csv", index=False)

    rtf = np.mean(rtf_list)
    gt_mos, syn_mos = map(np.mean, zip(*mos_list))
    with open(out_dir / "stats.txt", "w") as f:
        f.write(f"RTF: {rtf} (calculated by {len(test_ds)} files)\n")
        f.write(f"UTMOS-GT: {gt_mos} (calculated by {len(test_ds)} files)\n")
        f.write(f"UTMOS-Syn: {syn_mos} (calculated by {len(test_ds)} files)\n")


if __name__ == "__main__":
    main()
