import argparse
import logging

from sklearn.metrics.cluster import adjusted_rand_score, rand_score, adjusted_mutual_info_score
from lhotse import load_manifest_lazy

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
        "--field-1",
        type=str,
        required=True,
        help="The field name of the clustering labels"
    )
    
    parser.add_argument(
        "--field-2",
        type=str,
        required=True,
    )
    
    return parser.parse_args()

def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 30.0:
        return False
    return True

def compare_clusterings(
    manifest_1: str,
    manifest_2: str,
    field_1: str,
    field_2: str,
):
    cuts_1 = load_manifest_lazy(manifest_1)
    cuts_2 = load_manifest_lazy(manifest_2)
    
    cuts_1 = cuts_1.filter(remove_short_and_long_utt)
    cuts_2 = cuts_2.filter(remove_short_and_long_utt)
    
    cuts_1 = cuts_1.sort_like(cuts_2)
    
    clusters_1 = []
    clusters_2 = []
    for c1, c2 in zip(cuts_1, cuts_2):
        cluster_1 = getattr(c1, f"{field_1}_cluster")
        cluster_2 = getattr(c2, f"{field_2}_cluster")
        
        if len(cluster_1) != len(cluster_2):
            min_len = min(len(cluster_1), len(cluster_2))
            cluster_1 = cluster_1[:min_len]
            cluster_2 = cluster_2[:min_len]  
            
        clusters_1 += cluster_1
        clusters_2 += cluster_2
    
    logging.info(f"A total of {len(clusters_1)} samples")
    ri = rand_score(clusters_1, clusters_2)
    ari = adjusted_rand_score(clusters_1, clusters_2)
    ami = adjusted_mutual_info_score(clusters_1, clusters_2)
    logging.info(f"RI: {ri}, ARI: {ari}, AMI: {ami}")
        
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_parser()
    
    compare_clusterings(
        manifest_1=args.manifest_1,
        manifest_2=args.manifest_2,
        field_1=args.field_1,
        field_2=args.field_2,
    )