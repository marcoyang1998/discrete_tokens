import argparse
import joblib
import logging
import os

import torch
from lhotse import load_manifest_lazy, CutSet
from lhotse.utils import fastcopy
import numpy as np
from scipy.optimize import linear_sum_assignment

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
        "--topk",
        type=int,
        default=4,
    )
    
    parser.add_argument(
        "--cluster-mapping-path",
        type=str,
        required=True,
    )
    return parser.parse_args()

def maximize_label_agreement(labels1, labels2, n_clusters):
    # Step 1: Create confusion matrix
    confusion_matrix = np.zeros((n_clusters, n_clusters), dtype=int)
    for i in range(len(labels1)):
        confusion_matrix[labels1[i], labels2[i]] += 1

    # Step 2: Apply Hungarian algorithm on the negative of the confusion matrix
    row_ind, col_ind = linear_sum_assignment(-confusion_matrix)

    # Step 3: Create a dictionary mapping old labels to new labels
    label_mapping = {row: col for row, col in zip(row_ind, col_ind)}

    # Step 4: Map labels in the first model to match the second model's clusters
    new_labels1 = np.array([label_mapping[label] for label in labels1])

    return new_labels1, label_mapping, confusion_matrix
    

def merge_clusters(args):
    manifest_1 = args.manifest_1
    manifest_2 = args.manifest_2
    output_manifest = args.output_manifest
    topk = args.topk
    cluster_mapping_path = args.cluster_mapping_path
    
    cuts_1 = load_manifest_lazy(manifest_1)
    cuts_2 = load_manifest_lazy(manifest_2)
    
    cuts_2 = cuts_2.sort_like(cuts_1)
    
    # Obtain the cluster mapping
    # This is done by constructing the confusion matrix between the token labels of the two kmeans clusters
    # For each label in cluster 1, we find the top-k most frequent labels in cluster 2, and combine the rest
    # labels as a single label. In the end, we will have n_clusters * (top_k + 1) new clusters after merging
    if not os.path.exists(cluster_mapping_path):
        clusters_1 = []
        clusters_2 = []
        for cut1, cut2 in zip(cuts_1, cuts_2):
            c1 = getattr(cut1, "tokens")
            c2 = getattr(cut2, "tokens")
            
            if len(c1) != len(c2):
                min_len = min(len(c1), len(c2))
                c1 = c1[:min_len]
                c2 = c2[:min_len]  
                
            clusters_1 += c1
            clusters_2 += c2
        
        logging.info(f"A total of {len(clusters_1)} samples")
        
        n_clusters = 500
        new_labels, label_mapping, confusion_matrix = maximize_label_agreement(clusters_1, clusters_2, n_clusters)
        
        values, indices = torch.from_numpy(confusion_matrix).topk(topk, dim=1)
        new_km_indices = torch.arange(n_clusters).repeat(n_clusters, 1).T * (topk+1)
        for i in range(n_clusters):
            cur_indices = indices[i]
            new_km_indices[i, cur_indices] = torch.arange(1,topk+1) + (topk+1) * i
        
        assert new_km_indices.unique().numel() == n_clusters * (topk+1)
        logging.info(f"Saving the cluster mapping to {cluster_mapping_path}")
        torch.save(new_km_indices, cluster_mapping_path)
    else:
        logging.info(f"Loading the cluster mapping from {cluster_mapping_path}")
        new_km_indices = torch.load(cluster_mapping_path)
    
    new_cuts = []
    count = 0
    for cut1, cut2 in zip(cuts_1, cuts_2):
        c1 = getattr(cut1, "tokens")
        c2 = getattr(cut2, "tokens")
        
        if len(c1) != len(c2):
            min_len = min(len(c1), len(c2))
            c1 = c1[:min_len]
            c2 = c2[:min_len]
        
        new_tokens = []
        for i,j in zip(c1, c2):
            new_tokens.append(new_km_indices[i,j].item())
        
        new_cut = fastcopy(
            cut1,
            custom={"tokens": new_tokens}
        )
        new_cuts.append(new_cut)
        count += 1
        if count % 200 == 0:
            logging.info(f"Processed {count} cuts")
        
    new_cuts = CutSet.from_cuts(new_cuts)
    logging.info(f"Saving to {output_manifest}")
    new_cuts.to_jsonl(output_manifest)
        
    
if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    merge_clusters(args)
        
        
    
        
    