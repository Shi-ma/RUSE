#!/usr/bin/env bash

SR_MODEL=$1
TSV_PATH=$2

NPZ_OUT_DIR='../npz'

MODE='test'
CASE='true'

python ../src/tsv2npz.py --mode ${MODE} --tsv_path ${TSV_PATH} --npz_out_dir ${NPZ_OUT_DIR} --sr_model ${SR_MODEL} --case ${CASE}
