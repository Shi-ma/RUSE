#!/usr/bin/env bash

SR_MODEL=$1
TSV_PATH=$2

NPZ_OUT_DIR='../npz'

MODE='test'
CASE='true'
IS_DIR='../encoder_models/InferSent'
QT_DIR='../encoder_models/S2V'
QT_PRETRAINED_DIR='../encoder_models/S2V/pretrained_models'

python ../src/tsv2npz.py --mode ${MODE} --tsv_path ${TSV_PATH} --npz_out_dir ${NPZ_OUT_DIR} --sr_model ${SR_MODEL} --case ${CASE} --is_dir ${IS_DIR} --qt_dir ${QT_DIR} --qt_pretrained_dir ${QT_PRETRAINED_DIR}
