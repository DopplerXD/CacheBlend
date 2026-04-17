#!/usr/bin/env bash

# 1. Cmrc: Qwen
python example/blend_cmrc.py \
  --model-name /root/models/Qwen2.5-1.5B \
  --count 50 \
  --qaw-ratio 0.7

# 2. CMRC ratio curve
python example/blend_curve.py \
  --dataset cmrc \
  --model-name /root/models/Qwen2.5-1.5B \
  --count 50 \
  --qaw-ratio-min 0 \
  --qaw-ratio-max 1 \
  --qaw-ratio-step 0.05 \
  --suffix-len 32

# 3. CMRC suffix_len curve
python example/blend_suffix_curve.py \
  --dataset cmrc \
  --model-name /root/models/Qwen2.5-1.5B \
  --count 50 \
  --qaw-ratio 0.7 \
  --suffix-len-min 0 \
  --suffix-len-max 32 \
  --suffix-len-step 8

