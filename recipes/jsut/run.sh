#!/bin/bash

bin_dir=../../src/convnext_tts/bin

HYDRA_FULL_ERROR=1 python $bin_dir/preprocess.py
HYDRA_FULL_ERROR=1 python $bin_dir/train.py