# convnext-tts

Unofficial implementation of ConvNeXt-TTS([paper](https://ieeexplore.ieee.org/document/10446890)) for my experiment.  
The model architecture has been slightly modified.

# Usage

## 0. Setup environment
Install dependencies using uv([install instructions](https://docs.astral.sh/uv/getting-started/installation/)).
```
$ uv sync
```

Now I use wandb for logger, you have to login with wandb.
```
$ wandb login
```

## 1. Download dataset
Download JSUT corpus([link](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)) and fullcontext label([link](https://github.com/sarulab-speech/jsut-label)) and then sample wave files(basic5000) to 24kHz.

## 2. Change configuration 
Create a `default.yaml`(default setting) file under the `convnext_tts/bin/conf/path`directory, setting `wav_dir`, `lab_dir` and `data_root` according to your environment, using `src/convnext_tts/bin/conf/path/dummy.yaml` as a reference.  
If you wanna train original dataset, please change the mel-spectrogram configuration under `src/convnext_tts/bin/conf/mel/default.yaml`.

## 3. Training and synthesis (e.g. `exp/jsut`)
### 3.0. (Optional) 00_prepare
This is optional since it was prepared only for my environment.

### 3.1. Preprocessing
Extract F0 and generate train/valid dataframes.
```
$ cd exp/jsut/01_preprocess
$ ./run.sh
```

### 3.2. Training
Supported models:
- 02_default : WaveNeXt vocoder
- 03_vocos : Vocos vocoder

```
$ cd exp/jsut/{EXP_NAME}/01_train
$ ./run.sh
```

### 3.3. Synthesis
Generate samples with RTF calculation and UTMOS evaluation.

```
$ cd exp/jsut/{EXP_NAME}/02_syn
$ ./run.sh
```

I found that training WaveNeXt is slower than other vocoder models because the model must adjust the edges of each frame in the last linear layer.  
Therefore, when using WaveNeXt for end-to-end learning, the discriminator may become too strong in the early stages of training, so it might be necessary to perform some preliminary training beforehand like JETS.  
Please note that this issue may be caused by my implementation or data, and therefore it should not be generalized.

The generated samples (trained by exp/jsut/03_vocos) are of poor quality but are barely audible...

