#!/usr/bin/env bash

model_name=whisper
model_version=turbo
n_clusters=2000
layer_idx=28

fbank_dir=data/fbank_${model_name}_${model_version}_km${n_clusters}_normalized
mkdir -p $fbank_dir

kmeans_model=kmeans_model_normalized/${model_name}-${model_version}-layer-${layer_idx}.kmeans.${n_clusters}.model

for subset in balanced; do
    manifest_path=manifests/${subset}-${model_name}-${model_version}-layer-${layer_idx}.jsonl.gz
    output_manifest_path=$fbank_dir/cuts_audioset_${subset}.jsonl.gz

    python train_kmeans.py \
        --model-name $model_name \
        --model-version $model_version \
        --manifest-path $manifest_path \
        --n-clusters $n_clusters \
        --normalize True \
        --kmeans-model-path $kmeans_model \
        --output-manifest $output_manifest_path
done