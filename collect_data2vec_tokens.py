import argparse
import joblib
import logging
import os

import torch
import numpy as np
from lhotse import load_manifest_lazy, CutSet
from lhotse.cut import MonoCut
from lhotse.utils import fastcopy
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from models import Data2Vec
from train_kmeans import normalize_embedding


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model-version",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="The index starts from 1, so if you want the 12-th layer feature, just set it to 12"
    )
    
    parser.add_argument(
        "--manifest-path",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--output-manifest-path",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--kmeans-model",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--max-duration",
        type=int,
        default=200,
    )
    
    return parser.parse_args()       

def prepare_audios(batch):
    audio_pt = batch["audio"]
    audio_lens_pt = batch["audio_lens"]
    
    audios = []
    for i in range(audio_pt.shape[0]):
        audios.append(audio_pt[i, :audio_lens_pt[i]].numpy())
    return audios

@torch.no_grad()
def collect_tokens(
    model_version,
    manifest_path,
    kmeans_model_path,
    output_manifest_path,
    layer_idx=21,
    max_duration=200
):
    # loading the multi teacher model
    model = Data2Vec(model_version=model_version)
    model.eval()
    
    device = torch.device("cuda")
    model.to(device)
    
    manifest = load_manifest_lazy(manifest_path)
    dataset = UnsupervisedWaveformDataset(
        manifest
    )
    
    sampler = DynamicBucketingSampler(
        manifest,
        max_duration=max_duration,
        shuffle=False,
        drop_last=False,
    )
    
    dl = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=None,
        num_workers=1,
        persistent_workers=False,
    )
    
    device = torch.device("cuda")
    model.to(device)
    
    # load the normalization stats
    logging.info("Loading normalization stats")
    data2vec_mean = np.load("normalization_stats/data2vec-large-mu.npy")
    data2vec_std = np.load("normalization_stats/data2vec-large-std.npy")
    
    # load the kmeans model
    logging.info(f"Loading kmeans model from {kmeans_model_path}")
    kmeans_model = joblib.load(kmeans_model_path)
    
    new_cuts = []
    count = 0
    # extract the kmeans label
    for i, batch in enumerate(dl):
        cuts = batch["cuts"]
        audios = prepare_audios(batch)
        d2v_embeddings, embedding_lens = model.extract_features(audios, layer_idx)
        
        for j, cut in enumerate(cuts):
            cut = cut if isinstance(cut, MonoCut) else cut.tracks[0].cut
            cur_d2v_embedding = normalize_embedding(
                d2v_embeddings[j, :embedding_lens[j], :],
                data2vec_mean,
                data2vec_std
            )[0]
            
            labels = kmeans_model.predict(cur_d2v_embedding)
            
            new_cut = fastcopy(
                cut,
                custom = {"tokens": labels.tolist()},
            )
            new_cuts.append(new_cut)
            count += 1
            if count % 200 == 0:
                logging.info(f"Processed {count} cuts.")
                
    new_cuts = CutSet.from_cuts(new_cuts)
    logging.info(f"Saving the manifest to {output_manifest_path}")
    new_cuts.to_jsonl(output_manifest_path)
            
                
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    
    if os.path.exists(args.output_manifest_path):
        logging.info(f"The manifest {args.output_manifest_path} already exists. Skip this subset.")
    else:
        collect_tokens(
            model_version=args.model_version,
            manifest_path=args.manifest_path,
            kmeans_model_path=args.kmeans_model,
            output_manifest_path=args.output_manifest_path,
            layer_idx=args.layer_idx,
            max_duration=args.max_duration,
        )