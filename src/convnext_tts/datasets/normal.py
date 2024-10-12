from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from convnext_tts.frontend.ja import text_to_sequence
from convnext_tts.transforms.mel import MelSpectrogramTransform


class NormalDataset(Dataset):
    def __init__(
        self,
        df_file: str,
        wav_dir: str,
        cf0_dir: str,
        vuv_dir: str,
        to_mel: MelSpectrogramTransform,
    ):
        self.wav_dir = Path(wav_dir)
        self.cf0_dir = Path(cf0_dir)
        self.vuv_dir = Path(vuv_dir)
        self.to_mel = to_mel

        df = pd.read_csv(df_file)
        # convert phone string separated space(" ") into list
        df.iloc[:, 1] = df.iloc[:, 1].str.split()
        self.data = df.values.tolist()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        fname, phoneme = self.data[idx]
        bname = Path(fname).stem

        phoneme = text_to_sequence(phoneme)
        phoneme = torch.tensor(phoneme, dtype=torch.long)

        wav_file = self.wav_dir / f"{bname}.wav"
        cf0_file = self.cf0_dir / f"{bname}.npy"
        vuv_file = self.vuv_dir / f"{bname}.npy"

        wav, _ = torchaudio.load(wav_file)
        mel = self.to_mel(wav).squeeze(0)

        cf0 = torch.tensor(np.load(cf0_file), dtype=torch.float)[: mel.size(-1)]
        vuv = torch.tensor(np.load(vuv_file), dtype=torch.float)[: mel.size(-1)]
        return bname, phoneme, mel, cf0, vuv, wav

    def num_tokens(self, idx):
        return len(self.data[idx][1])

    def ordered_indices(self):
        lengths = np.array([len(x[1]) for x in self.data])
        indices = np.random.permutation(len(self))
        indices = indices[np.argsort(np.array(lengths)[indices], kind="mergesort")]
        return indices


class NormalCollator:
    def __call__(self, batch):
        # I'm not sure whether using `torch.nn.utils.rnn.pad_sequence` is good or if the following method is better
        (bnames, phonemes, mels, cf0s, vuvs, wavs) = zip(*batch)

        B = len(bnames)
        phone_lengths = [x.size(-1) for x in phonemes]
        frame_lengths = [x.size(-1) for x in mels]
        sample_lengths = [x.size(-1) for x in wavs]

        phone_max_length = max(phone_lengths)
        frame_max_length = max(frame_lengths)
        sample_max_length = max(sample_lengths)
        mel_dim = mels[0].size(0)

        phoneme_pad = torch.zeros(size=(B, phone_max_length), dtype=torch.long)
        mel_pad = torch.zeros(size=(B, mel_dim, frame_max_length), dtype=torch.float)
        cf0_pad = torch.zeros(size=(B, 1, frame_max_length), dtype=torch.float)
        vuv_pad = torch.zeros(size=(B, 1, frame_max_length), dtype=torch.float)
        wav_pad = torch.zeros(size=(B, 1, sample_max_length), dtype=torch.float)
        for i in range(B):
            p_l, f_l, s_l = phone_lengths[i], frame_lengths[i], sample_lengths[i]
            phoneme_pad[i, :p_l] = phonemes[i]
            mel_pad[i, :, :f_l] = mels[i]
            cf0_pad[i, :, :f_l] = cf0s[i]
            vuv_pad[i, :, :f_l] = vuvs[i]
            wav_pad[i, :, :s_l] = wavs[i]

        phone_lengths = torch.LongTensor(phone_lengths)
        frame_lengths = torch.LongTensor(frame_lengths)
        sample_lengths = torch.LongTensor(sample_lengths)

        return (
            bnames,
            phoneme_pad,
            mel_pad,
            cf0_pad,
            vuv_pad,
            wav_pad,
            phone_lengths,
            frame_lengths,
            sample_lengths,
        )
