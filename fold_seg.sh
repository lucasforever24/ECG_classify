#!/usr/bin/env bash

source activate py37

python run_fold_seg_pipeline.py \
> log/fold_seg_4_class_1
