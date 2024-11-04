import argparse
import logging
import joblib
import os

import torch
from lhotse import load_manifest_lazy, CutSet
from lhotse.utils import fastcopy
import numpy as np

from train_kmeans import get_km_model, normalize_embedding
from icefall.utils import str2bool

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
        "--model-1",
        type=str,
    )
    
    parser.add_argument(
        "--model-2",
        type=str,
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
    
    parser.add_argument(
        "--normalize",
        type=str2bool,
        default=False,
        help="If normalize each dimension to zero mean and unit variance"
    )
    
    return parser.parse_args()

def train_kmeans(args, cuts_1, cuts_2):
    # train the kmeans model if not exists
    all_embeddings_1 = []
    all_embeddings_2 = []
    count = 0
    for c1, c2 in zip(cuts_1, cuts_2):
        embed_1 = c1.load_custom(f"{args.model_1}_embedding")
        embed_2 = c2.load_custom(f"{args.model_2}_embedding")
        if embed_1.shape[0] != embed_2.shape[0]:
            min_len = min(embed_1.shape[0], embed_2.shape[0])
            embed_1 = embed_1[:min_len]
            embed_2 = embed_2[:min_len]
        all_embeddings_1.append(embed_1)
        all_embeddings_2.append(embed_2)
        
        count += 1
        if count % 1000 == 0:
            logging.info(f"Cuts processed until now: {count}")

    all_embeddings_1 = np.concatenate(all_embeddings_1)
    all_embeddings_2 = np.concatenate(all_embeddings_2)
    if args.normalize:
        global_mean_1 = np.load(f"normalization_stats/{args.model_1}-large-mu.npy")
        global_mean_2 = np.load(f"normalization_stats/{args.model_2}-large-mu.npy")
        global_std_1 = np.load(f"normalization_stats/{args.model_1}-large-std.npy")
        global_std_2 = np.load(f"normalization_stats/{args.model_2}-large-std.npy")
        all_embeddings_1 = normalize_embedding(all_embeddings_1, global_mean_1, global_std_1)[0]
        all_embeddings_2 = normalize_embedding(all_embeddings_2, global_mean_2, global_std_2)[0]
    
    if args.merge_strategy == "avg":
        all_embeddings = (all_embeddings_1 + all_embeddings_2) * 0.5
    elif args.merge_strategy == "concat":
        all_embeddings = np.concatenate((all_embeddings_1, all_embeddings_2), axis=-1)

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
    
    return km_model

def compute_kmeans_label(args):
    cuts_1 = load_manifest_lazy(args.manifest_1)
    cuts_2 = load_manifest_lazy(args.manifest_2)
    
    cuts_2 = cuts_2.sort_like(cuts_1)
        
    if os.path.exists(args.kmeans_model_path):
        logging.info(f"Found a pre-trained kmeans model. Loading it from {args.kmeans_model_path}")
        km_model = joblib.load(args.kmeans_model_path)
    else:
        logging.info(f"Start training kmeans model on the merged embeddings")
        km_model = train_kmeans(args, cuts_1, cuts_2)
    
    if args.normalize:
        global_mean_1 = np.load(f"normalization_stats/{args.model_1}-large-mu.npy")
        global_mean_2 = np.load(f"normalization_stats/{args.model_2}-large-mu.npy")
        global_std_1 = np.load(f"normalization_stats/{args.model_1}-large-std.npy")
        global_std_2 = np.load(f"normalization_stats/{args.model_2}-large-std.npy")
    
    new_cuts = []
    count = 0
    
    for c1, c2 in zip(cuts_1, cuts_2):
        embed_1 = c1.load_custom("hubert_embedding")
        embed_2 = c2.load_custom("data2vec_embedding")
        if embed_1.shape[0] != embed_2.shape[0]:
            min_len = min(embed_1.shape[0], embed_2.shape[0])
            embed_1 = embed_1[:min_len]
            embed_2 = embed_2[:min_len]
        if args.normalize:
            embed_1 = normalize_embedding(embed_1, global_mean_1, global_std_1)[0]
            embed_2 = normalize_embedding(embed_2, global_mean_2, global_std_2)[0]
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
    compute_kmeans_label(args)
        
    
    