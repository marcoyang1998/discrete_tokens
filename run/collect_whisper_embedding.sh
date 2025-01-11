#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES=2

model_name=whisper
model_version=turbo
layer_idx=-1

python collect_embeddings.py \
    --model-name $model_name \
    --model-version $model_version \
    --layer-idx $layer_idx \
    --subset aishell_subset \
    --max-duration 200 \
    --input-manifest /cpfs01/shared/speechllm/audio_encoder/icefall_mvq_quantizer/egs/librispeech/ASR/data/manifests/aishell_subset.jsonl.gz