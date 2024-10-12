from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import pyworld as pw
import soundfile as sf
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from convnext_tts.frontend.ja import pp_symbols
from convnext_tts.utils.logging import logger
from convnext_tts.utils.tqdm import tqdm_joblib


@hydra.main(config_path="conf", version_base=None, config_name="config")
def main(cfg):
    wav_dir = Path(cfg.path.wav_dir)
    lab_dir = Path(cfg.path.lab_dir)

    data_root = Path(cfg.path.data_root)
    if data_root.exists() and not cfg.preprocess.overwrite:
        logger.info("Skip preprocessing")
        return
    # Create directories
    [
        Path(d).mkdir(parents=True, exist_ok=False)
        for d in [cfg.path.df_dir, cfg.path.cf0_dir, cfg.path.vuv_dir]
    ]

    logger.info("Start Processing...")
    wav_files = list(sorted(wav_dir.glob("*.wav")))
    lab_files = list(sorted(lab_dir.glob("*.lab")))

    assert len(wav_files) == len(lab_files), "Number of files mismatch"

    logger.info("Extracting phonems from fullcontext label...")
    phonemes = []
    for lab_file in tqdm(lab_files):
        with open(lab_file) as f:
            labels = [line.strip() for line in f]
            phoneme = " ".join(pp_symbols(labels))
            phonemes.append(phoneme)
    assert len(wav_files) == len(phonemes), "Number of files mismatch"

    wav_names = [f.stem for f in wav_files]
    df = pd.DataFrame({"wav": wav_names, "phonemes": phonemes})
    valid_size = int(len(df) * 0.02)
    train_df = df.iloc[valid_size:]
    valid_df = df.iloc[:valid_size]
    train_df.to_csv(cfg.path.train_df_file, index=False)
    valid_df.to_csv(cfg.path.valid_df_file, index=False)
    logger.info("Saved train and valid dataframe")

    logger.info("Extracting f0, cf0, and vuv...")

    def _process(wav_file):
        bname = wav_file.stem
        wav, sr = sf.read(wav_file)
        assert sr == cfg.mel.sample_rate
        f0, _ = pw.harvest(
            wav, sr, frame_period=cfg.mel.hop_length / cfg.mel.sample_rate * 1e3
        )
        vuv = (f0 != 0).astype(np.float32)
        # linear interpolation
        x = np.arange(len(f0))
        idx = np.nonzero(f0)
        cf0 = np.interp(x, x[idx], f0[idx])
        np.save(f"{cfg.path.cf0_dir}/{bname}.npy", cf0)
        np.save(f"{cfg.path.vuv_dir}/{bname}.npy", vuv)

    with tqdm_joblib(len(wav_files)):
        Parallel(n_jobs=cfg.preprocess.n_jobs)(delayed(_process)(f) for f in wav_files)


if __name__ == "__main__":
    main()
