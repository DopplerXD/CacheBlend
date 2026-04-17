#!/usr/bin/env bash

# 1. MusiQue: full_prefill + qaw_default
python example/blend_runner.py \
  --dataset musique \
  --model-name /root/models/Qwen2.5-1.5B \
  --count 50 \
  --qaw-ratio 0.7 \
  --suffix-len 32 \
  --methods prefill_qaw

# 2. WikiMQA: full_prefill + qaw_default
python example/blend_runner.py \
  --dataset wikimqa \
  --model-name /root/models/Qwen2.5-1.5B \
  --count 50 \
  --qaw-ratio 0.7 \
  --suffix-len 32 \
  --methods prefill_qaw

# 3. CMRC: full_prefill + qaw_default
python example/blend_cmrc.py \
  --model-name /root/models/Qwen2.5-1.5B

# 4. CMRC ratio curve
python example/blend_curve.py \
  --dataset cmrc \
  --model-name /root/models/Yi-6B \
  --count 50 \
  --qaw-ratio-min 0 \
  --qaw-ratio-max 1 \
  --qaw-ratio-step 0.05 \
  --suffix-len 32

# 5. CMRC suffix_len curve
python example/blend_suffix_curve.py \
  --dataset cmrc \
  --model-name /root/models/Yi-6B \
  --count 50 \
  --qaw-ratio 0.7 \
  --suffix-len-min 0 \
  --suffix-len-max 32 \
  --suffix-len-step 8

