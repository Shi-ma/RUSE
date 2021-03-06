#!/usr/bin/env bash

SR_MODEL=$1
NPZ_DIR=$2

GPU=-1

if [ ${SR_MODEL} = 'IS' ]; then
  LAYER=3
  UNIT=4096
  BATCH=512
  DROP=0.1
elif [ ${SR_MODEL} = 'QT' ]; then
  LAYER=2
  UNIT=4096
  BATCH=1024
  DROP=0.5
elif [ ${SR_MODEL} = 'USE' ]; then
  LAYER=2
  UNIT=4096
  BATCH=512
  DROP=0.5
elif [ ${SR_MODEL} = 'IS_QT_USE' ]; then
  LAYER=2
  UNIT=2048
  BATCH=128
  DROP=0.3
fi

python ../src/make_score.py --npz_dir ${NPZ_DIR} --SR_models ${SR_MODEL} -l ${LAYER} -u ${UNIT} -b ${BATCH} -dr ${DROP} --gpu ${GPU}
