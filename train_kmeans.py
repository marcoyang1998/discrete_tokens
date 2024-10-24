import argparse
import logging
import joblib
import os

from lhotse import CutSet, load_manifest_lazy
from lhotse.cut import MonoCut

import numpy as np
from sklearn.cluster import MiniBatchKMeans

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--manifest-path",
        type=str,
        required=True,
    )
    
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
    parser.add_argument("--output-manifest", type=str, required=True)
    
    return parser.parse_args()


def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters=n_clusters,
        init=init,
        max_iter=max_iter,
        batch_size=batch_size,
        verbose=1,
        compute_labels=False,
        tol=tol,
        max_no_improvement=max_no_improvement,
        init_size=None,
        n_init=n_init,
        reassignment_ratio=reassignment_ratio,
    )

def train_kmeans(args, cuts):
    # train a kmeans model and return it
    all_embeddings = []
    for i, cut in enumerate(cuts):
        embedding = cut.load_custom("wavlm_embedding")
        all_embeddings.append(embedding)
        if i % 200 == 0 and i > 0:
            logging.info(f"Loaded {i} cuts")
    
    logging.info("Finish loading all the wavlm embeddings")
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
    import pdb; pdb.set_trace()
    km_model.fit(all_embeddings)
    
    logging.info(f"Saving the kmeans model to {args.kmeans_model_path}")
    joblib.dump(km_model, args.kmeans_model_path)
    
    return km_model

def compute_kmeans_label(args):
    
    cuts = load_manifest_lazy(args.manifest_path)
    
    # train a kmeans model
    if not os.path.exists(args.kmeans_model_path):
        km_model = train_kmeans(args, cuts=cuts)
    else:
        logging.info(f"Loading pretrained kmeans model from {args.kmeans_model_path}")
        km_model = joblib.load(args.kmeans_model_path)
    
    new_cuts = []
    for i, cut in enumerate(cuts):
        new_cut = MonoCut(
            id=cut.id,
            start=cut.start,
            duration=cut.duration,
            channel=cut.channel,
        )
        embedding = cut.load_custom("wavlm_embedding")
        labels = km_model.predict(embedding)
        new_cut.wavlm_cluster = labels.tolist()
        
        new_cuts.append(new_cut)
        if i % 200 == 0 and i > 0:
            logging.info(f"Processed {i} cuts")
        
    CutSet.from_cuts(new_cuts).to_jsonl(args.output_manifest)
    logging.info(f"The output manifest is saved to {args.output_manifest}")
        
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    compute_kmeans_label(args)