#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=2

model_name=whisper
model_version=turbo
n_clusters=2000
layer_idx=28

fbank_dir=data/fbank_${model_name}_${model_version}_km${n_clusters}_normalized
global_mean_file=normalization_stats/${model_name}-${model_version}-mu.npy
global_std_file=normalization_stats/${model_name}-${model_version}-std.npy
kmeans_model=kmeans_model_normalized/${model_name}-${model_version}-layer-${layer_idx}.kmeans.${n_clusters}.model

for subset in balanced; do
    python collect_kmeans_tokens.py \
        --model-name $model_name \
        --model-version $model_version \
        --manifest-path data/fbank_whisper_turbo_km2000_normalized/cuts_audioset_balanced.jsonl.gz \
        --output-manifest-path $fbank_dir/cuts_audioset_${subset}.jsonl.gz \
        --normalize True \
        --layer-idx $layer_idx \
        --global-mean-file $global_mean_file \
        --global-std-file $global_std_file \
        --weighted-combine False \
        --kmeans-model $kmeans_model \
        --max-duration 500 
done