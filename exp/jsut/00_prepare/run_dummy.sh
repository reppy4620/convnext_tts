#!/bin/bash

wav_dir=/path/to/jsut_wav_24k
lab_dir=/path/to/jsut-label/labels/basic5000

out_dir=/path/to/data_root
mkdir -p ${out_dir}

rsync -avu ${wav_dir}/ ${out_dir}/wav/
rsync -avu ${lab_dir}/ ${out_dir}/lab/
