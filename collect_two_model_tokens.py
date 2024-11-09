import argparse
import joblib
import logging
import os

import torch
import numpy as np
from lhotse import load_manifest_lazy, CutSet
from lhotse.utils import fastcopy
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset
from transformers import AutoProcessor, Data2VecAudioModel, AutoModel, Wav2Vec2FeatureExtractor

from models import Data2Vec, WavlmModel, HuBERT
from utils import make_pad_mask
from train_kmeans import normalize_embedding

MODEL_DICT = {
    "wavlm": WavlmModel,
    "data2vec": Data2Vec,
    "hubert": HuBERT,
}

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model-1",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--model-2",
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

class MultiTeacherModel(torch.nn.Module):
    def __init__(
        self, 
        model_1: str,
        model_2: str,
    ):
        super().__init__()
        self.model_1 = MODEL_DICT[model_1]()
        self.model_2 = MODEL_DICT[model_2]()
        
    def extract_features(
        self,
        batch,
        layer_idx,
    ):
        batch_size = len(batch["cuts"])
        embeddings_1, embedding_lens_1 = self.model_1.extract_features(batch, layer_idx)
        embedding_list_1 = []
        for i in range(batch_size):
            cur_embedding = embeddings_1[i, :embedding_lens_1[i]]
            embedding_list_1.append(cur_embedding)
        
        embeddings_2, embedding_lens_2 = self.model_2.extract_features(batch, layer_idx)
        mbedding_list_2 = []
        for i in range(batch_size):
            cur_embedding = embeddings_2[i, :embedding_lens_2[i]]
            mbedding_list_2.append(cur_embedding)
        
        return embedding_list_1, mbedding_list_2
    
        
@torch.no_grad()
def collect_tokens(
    model_1,
    model_2,
    manifest_path,
    kmeans_model_path,
    output_manifest_path,
    layer_idx=21,
    max_duration=200
):
    # loading the multi teacher model
    model = MultiTeacherModel(
        model_1=model_1,
        model_2=model_2,
    )
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
    mean_1 = np.load(f"normalization_stats/{model_1}-large-mu.npy")
    std_1 = np.load(f"normalization_stats/{model_1}-large-std.npy")
    mean_2 = np.load(f"normalization_stats/{model_2}-large-mu.npy")
    std_2 = np.load(f"normalization_stats/{model_2}-large-std.npy")
    
    # load the kmeans model
    logging.info(f"Loading kmeans model from {kmeans_model_path}")
    kmeans_model = joblib.load(kmeans_model_path)
    
    new_cuts = []
    count = 0

    # extract the kmeans label
    for i, batch in enumerate(dl):
        cuts = batch["cuts"]
        embeddings_1, embeddings_2 = model.extract_features(batch, layer_idx)
        
        for j, cut in enumerate(cuts):
            cur_embedding_1 = normalize_embedding(embeddings_1[j], mean_1, std_1)[0]
            cur_embedding_2 = normalize_embedding(embeddings_2[j], mean_2, std_2)[0]
            if cur_embedding_1.shape[0] != cur_embedding_2.shape[0]:
                min_len = min(cur_embedding_1.shape[0], cur_embedding_2.shape[0])
                cur_embedding_1 = cur_embedding_1[:min_len]
                cur_embedding_2 = cur_embedding_2[:min_len]
            
            # concate along the feature dimension
            merged_embedding = np.concatenate((cur_embedding_1, cur_embedding_2), axis=-1)
            labels = kmeans_model.predict(merged_embedding)
            
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
            model_1=args.model_1,
            model_2=args.model_2,
            manifest_path=args.manifest_path,
            kmeans_model_path=args.kmeans_model,
            output_manifest_path=args.output_manifest_path,
            layer_idx=args.layer_idx,
            max_duration=args.max_duration,
        )