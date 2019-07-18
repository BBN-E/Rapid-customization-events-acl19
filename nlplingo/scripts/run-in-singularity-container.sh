#!/bin/bash
CDIR="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"

cmd="$@"

#SINGULARITY=/opt/singularity-2.3.1-x86_64_sl69_no_new_privs_set/bin/singularity
#SINGULARITY_IMAGE=/nfs/mercury-07/u35/jsilovsk/singularity/images/cuda9.0-cudnn7-devel-ubuntu16.04-tensorflow-gpu-1.6.0-py3-tensor2tensor.9e17755
SINGULARITY_ROOT="/usr/local"
LD_LIBRARY_PATH="$SINGULARITY_ROOT/lib:$LD_LIBRARY_PATH"
SINGULARITY_BIN="$SINGULARITY_ROOT/bin/singularity"
SINGULARITY_IMAGE="/nfs/raid87/u14/CauseEx/NN-events-requirements/singularity_image/nlplingo_v1.img"

cd $CDIR
env -i \
  SINGULARITYENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
  $SINGULARITY_BIN\
  exec \
  -B /nfs/raid87/u10 \
  -B /nfs/raid87/u11 \
  -B /nfs/raid87/u12 \
  -B /nfs/raid87/u13 \
  -B /nfs/raid87/u14 \
  -B /nfs/raid87/u15 \
  -B /nfs/raid84/u12 \
  -B /nfs/ld100/u10 \
  -B /nfs/raid66/u14 \
  -B /nfs/mercury-04/u40 \
  -B /nfs/mercury-04/u22 \
  --nv \
  $SINGULARITY_IMAGE $cmd

#  $SINGULARITY_IMAGE bash --norc

exit $?
