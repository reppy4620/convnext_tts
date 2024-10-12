# convnext-tts

Unofficial implementation of ConvNeXt-TTS([paper](https://ieeexplore.ieee.org/document/10446890)) for my experiment.  
The model architecture has been slightly modified.

# Usage

0. Install dependencies using uv([link](https://docs.astral.sh/uv/getting-started/installation/)).
1. Download JSUT corpus and fullcontext label([link](https://github.com/sarulab-speech/jsut-label)) and then sample wave files(basic5000) to 24kHz.
2. Create a `default.yaml` file under the `convnext_tts/bin/conf/path`directory, setting `wav_dir`, `lab_dir` and `data_root` according to your environment, using `src/convnext_tts/bin/conf/path/dummy.yaml` as a reference.
3. Run `exp/jsut/run.sh`.

I found that training WaveNeXt is slower than other vocoder models because the model must adjust the edges of each frame in the last linear layer.  
Therefore, when using WaveNeXt for end-to-end learning, the discriminator may become too strong in the early stages of training, so it might be necessary to perform some preliminary training beforehand like JETS.  
Please note that this issue may be caused by my implementation or data, and therefore it should not be generalized.

Still under development...