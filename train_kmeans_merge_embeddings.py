import argparse
import logging
import joblib
import os

import torch
from lhotse import load_manifest_lazy, CutSet
from lhotse.utils import fastcopy
import numpy as np

from train_kmeans import get_km_model

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--manifest-1",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--manifest-2",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--output-manifest",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--merge-strategy",
        type=str,
        choices=["avg", "concat"],
        required=True,
    )
    
    # clustering related arguments
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=500,
    )
    
    parser.add_argument(
        "--init", 
        default="k-means++", 
        type=str
    )
    
    parser.add_argument(
        "--max_iter", 
        default=100, 
        type=int
    )
    parser.add_argument(
        "--batch_size", 
        default=10000, 
        type=int
    )
    parser.add_argument("--max-no-improvement", default=100, type=int)
    parser.add_argument("--n-init", default=20, type=int)
    parser.add_argument("--reassignment-ratio", default=0.0, type=float)
    parser.add_argument("--kmeans-model-path", type=str, required=True)
    
    return parser.parse_args()

def train_km_merged_embeddings(args):
    cuts_1 = load_manifest_lazy(args.manifest_1)
    cuts_2 = load_manifest_lazy(args.manifest_2)
    
    cuts_2 = cuts_2.sort_like(cuts_1)
        
    # train the kmeans model if not exists
    if not os.path.exists(args.kmeans_model_path):
        logging.info("Start merging embeddings")
        all_embeddings = []
        count = 0
        for c1, c2 in zip(cuts_1, cuts_2):
            embed_1 = c1.load_custom("hubert_embedding")
            embed_2 = c2.load_custom("data2vec_embedding")
            if embed_1.shape[0] != embed_2.shape[0]:
                min_len = min(embed_1.shape[0], embed_2.shape[0])
                embed_1 = embed_1[:min_len]
                embed_2 = embed_2[:min_len]
            if args.merge_strategy == "avg":
                merged_embed = (embed_1 + embed_2) * 0.5
            elif args.merge_strategy == "concat":
                merged_embed = np.concatenate((embed_1, embed_2), axis=-1)
            
            all_embeddings.append(merged_embed)
            count += 1
            if count % 1000 == 0:
                logging.info(f"Cuts processed until now: {count}")
    
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        km_model = get_km_model(
            n_clusters=args.n_clusters,
            init=args.init,
            max_iter=args.max_iter,
            batch_size=args.batch_size,
            tol=0.0,
            max_no_improvement=args.max_no_improvement,
            n_init=args.n_init,
            reassignment_ratio=args.reassignment_ratio
        )
        
        logging.info("Start training the kmeans model")
        km_model.fit(all_embeddings)
        
        logging.info(f"Saving the kmeans model to {args.kmeans_model_path}")
        joblib.dump(km_model, args.kmeans_model_path)
    else:
        km_model = joblib.load(args.kmeans_model_path)
        logging.info(f"Loaded the kmeans model from {args.kmeans_model_path}")
        
    new_cuts = []
    count = 0
    
    for c1, c2 in zip(cuts_1, cuts_2):
        embed_1 = c1.load_custom("hubert_embedding")
        embed_2 = c2.load_custom("data2vec_embedding")
        if embed_1.shape[0] != embed_2.shape[0]:
            min_len = min(embed_1.shape[0], embed_2.shape[0])
            embed_1 = embed_1[:min_len]
            embed_2 = embed_2[:min_len]
        if args.merge_strategy == "avg":
            merged_embed = (embed_1 + embed_2) * 0.5
        elif args.merge_strategy == "concat":
            merged_embed = np.concatenate((embed_1, embed_2), axis=-1)
        tokens = km_model.predict(merged_embed)
        new_cut = fastcopy(
            c1,
            custom = {"tokens": tokens.tolist()},
        )
        
        count += 1
        new_cuts.append(new_cut)
        
        if count % 200 == 0 and count > 0:
            logging.info(f"Processed {count} cuts")
            
    new_cuts = CutSet.from_cuts(new_cuts)
    logging.info(f"Saving the new manifest to {args.output_manifest}")
    new_cuts.to_jsonl(args.output_manifest)
    
if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    train_km_merged_embeddings(args)
        
    
    