#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=7

model_name=whisper
model_version=turbo
n_clusters=2000
layer_idx=-1

# 1. Collect embeddings
for subset in mix; do
    python collect_embeddings.py \
        --model-name $model_name \
        --model-version $model_version \
        --layer-idx $layer_idx \
        --subset $subset \
        --input-manifest data/manifests_librispeech/librispeech_cuts_${subset}.jsonl.gz
done

# 2. train kmeans
fbank_dir=data/fbank_${model_name}_${model_version}_km${n_clusters}_normalized
mkdir -p $fbank_dir

global_mean_file=normalization_stats/${model_name}-${model_version}-libri-mu.npy
global_std_file=normalization_stats/${model_name}-${model_version}-libri-std.npy
kmeans_model=kmeans_model_normalized/${model_name}-${model_version}-layer-${layer_idx}.libri.kmeans.${n_clusters}.model

for subset in mix; do
    manifest_path=manifests/${subset}-${model_name}-${model_version}-layer-${layer_idx}.jsonl.gz
    output_manifest_path=$fbank_dir/librispeech_cuts_${subset}.jsonl.gz

    python train_kmeans.py \
        --model-name $model_name \
        --model-version $model_version \
        --manifest-path $manifest_path \
        --n-clusters $n_clusters \
        --normalize True \
        --global-mean-file $global_mean_file \
        --global-std-file $global_std_file \
        --kmeans-model-path $kmeans_model \
        --output-manifest $output_manifest_path
done

# 3. collect kmeans labels
for subset in dev-clean dev-other test-clean test-other; do
    python collect_kmeans_tokens.py \
        --model-name $model_name \
        --model-version $model_version \
        --manifest-path data/manifests_librispeech/librispeech_cuts_${subset}.jsonl.gz \
        --output-manifest-path $fbank_dir/librispeech_cuts_${subset}.jsonl.gz \
        --normalize True \
        --layer-idx $layer_idx \
        --global-mean-file $global_mean_file \
        --global-std-file $global_std_file \
        --weighted-combine False \
        --kmeans-model $kmeans_model \
        --max-duration 500 
done