# convnext-tts

Unofficial implementation of ConvNeXt-TTS([paper](https://ieeexplore.ieee.org/document/10446890)) for my experiment.


# Usage

0. Install dependencies using Rye([link](https://rye-up.com/guide/installation/)).
1. Download JSUT corpus and fullcontext label([link](https://github.com/sarulab-speech/jsut-label)) and then sample wave files(basic5000) to 24kHz.
2. Create a `default.yaml` file under the `convnext_tts/bin/conf/path`directory, setting `wav_dir`, `lab_dir` and `data_root` according to your environment, using `src/convnext_tts/bin/conf/path/dummy.yaml` as a reference.
3. Run `exp/jsut/run.sh`.

Now I'm running it, but it seems likely to fail....
