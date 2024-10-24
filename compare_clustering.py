import logging

from sklearn.metrics.cluster import adjusted_rand_score, rand_score, adjusted_mutual_info_score
from lhotse import load_manifest_lazy

def compare_clusterings(
    manifest_1,
    manifest_2,
):
    cuts_1 = load_manifest_lazy(manifest_1)
    cuts_2 = load_manifest_lazy(manifest_2)
    
    cuts_1 = cuts_1.sort_like(cuts_2)
    
    cluster1 = []
    cluster2 = []
    for c1, c2 in zip(cuts_1, cuts_2):
        wavlm_cluster = c1.wavlm_cluster
        hubert_cluster = c2.hubert_cluster
        
        assert len(wavlm_cluster)==len(hubert_cluster)
        cluster1 += wavlm_cluster
        cluster2 += hubert_cluster
    
    logging.info(f"A total of {len(cluster1)} samples")
    ri = rand_score(cluster1, cluster2)
    ari = adjusted_rand_score(cluster1, cluster2)
    ami = adjusted_mutual_info_score(cluster1, cluster2)
    logging.info(f"RI: {ri}, ARI: {ari}, AMI: {ami}")
        
    
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    
    
    manifest_1 = "manifests/dev-clean-wavlm-base-plus-layer--1-kmeans-label.jsonl.gz"
    manifest_2 = "manifests/dev-clean-hubert-base-layer-12-kmeans-label.jsonl.gz"
    compare_clusterings(
        manifest_1=manifest_1,
        manifest_2=manifest_2,
    )