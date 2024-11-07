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

from utils import make_pad_mask
from train_kmeans import normalize_embedding


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--hubert-ckpt",
        type=str,
        help="path to the hubert checkpoint",
        required=True,
    )
    
    parser.add_argument(
        "--data2vec-version",
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
        hubert_version,
        data2vec_version,
    ):
        super().__init__()
        self.hubert_processor = Wav2Vec2FeatureExtractor.from_pretrained(f"facebook/hubert-{hubert_version}-ll60k")
        self.hubert_model = AutoModel.from_pretrained(f"facebook/hubert-{hubert_version}-ll60k")
        
        self.d2v_processor = AutoProcessor.from_pretrained(f"facebook/data2vec-audio-{data2vec_version}")
        self.d2v_model = Data2VecAudioModel.from_pretrained(f"facebook/data2vec-audio-{data2vec_version}")
        
    def extract_hubert_features(self, batch, layer_idx):
        device = next(self.hubert_model.parameters()).device
        
        # prepare the input audio data
        audio_pt = batch["audio"]
        audio_lens_pt = batch["audio_lens"]
        audios = []
        for i in range(audio_pt.shape[0]):
            audios.append(audio_pt[i, :audio_lens_pt[i]].numpy())
            
        inputs = self.hubert_processor(
            audios, 
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)
        
        outputs = self.hubert_model(
            output_hidden_states=True,
            **inputs,
        )
        all_layer_results = outputs.hidden_states
        layer_results = all_layer_results[layer_idx].cpu().numpy()
        padding_mask = self.hubert_model._get_feature_vector_attention_mask(layer_results.shape[1], inputs["attention_mask"])
        embedding_lens = padding_mask.sum(dim=-1)
        
        return layer_results, embedding_lens
        
    def extract_data2vec_features(self, batch, layer_idx):
        device = next(self.d2v_model.parameters()).device
        
        # prepare the input audio data
        audio_pt = batch["audio"]
        audio_lens_pt = batch["audio_lens"]
        audios = []
        for i in range(audio_pt.shape[0]):
            audios.append(audio_pt[i, :audio_lens_pt[i]].numpy())
            
        inputs = self.d2v_processor(
            audios, 
            sampling_rate=16000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt"
        ).to(device)
        
        outputs = self.d2v_model(
            output_hidden_states=True,
            **inputs,
        )
        all_layer_results = outputs.hidden_states
        # If the model has 12 layers, it will return 13 features
        # with the first one as the input features after conv modules
        # So if you want to extract the 12-th layer's representation
        # Use layer_idx=12
        layer_results = all_layer_results[layer_idx].cpu().numpy()
        padding_mask = self.d2v_model._get_feature_vector_attention_mask(layer_results.shape[1], inputs["attention_mask"])
        embedding_lens = padding_mask.sum(dim=-1)
        
        return layer_results, embedding_lens
        
    
    def extract_features(self, batch, layer_idx):
        # extract features from hubert and data2vec models
        # the returned embeddings are a list of numpy arrays, whose padding
        # frames are discarded
        batch_size = len(batch["cuts"])
        
        hubert_embeddings, hubert_embedding_lens = self.extract_hubert_features(batch, layer_idx)
        hubert_embedding_list = []
        for i in range(batch_size):
            cur_embedding = hubert_embeddings[i, :hubert_embedding_lens[i]]
            hubert_embedding_list.append(cur_embedding)
            
        d2v_embeddings, d2v_embedding_lens = self.extract_data2vec_features(batch, layer_idx)
        d2v_embedding_list = []
        for i in range(batch_size):
            cur_embedding = d2v_embeddings[i, :d2v_embedding_lens[i]]
            d2v_embedding_list.append(cur_embedding)
        
        return hubert_embedding_list, d2v_embedding_list
        
        
@torch.no_grad()
def collect_tokens(
    hubert_ckpt,
    data2vec_version,
    manifest_path,
    kmeans_model_path,
    output_manifest_path,
    layer_idx=21,
    max_duration=200
):
    # loading the multi teacher model
    model = MultiTeacherModel(
        hubert_ckpt=hubert_ckpt,
        data2vec_version=data2vec_version,
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
    hubert_mean = np.load("normalization_stats/hubert-large-mu.npy")
    hubert_std = np.load("normalization_stats/hubert-large-std.npy")
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
        hubert_embeddings, d2v_embeddings = model.extract_features(batch, layer_idx)
        
        for j, cut in enumerate(cuts):
            cur_hubert_embedding = normalize_embedding(hubert_embeddings[j], hubert_mean, hubert_std)[0]
            cur_d2v_embedding = normalize_embedding(d2v_embeddings[j], data2vec_mean, data2vec_std)[0]
            if cur_hubert_embedding.shape[0] != cur_d2v_embedding.shape[0]:
                min_len = min(cur_hubert_embedding.shape[0], cur_d2v_embedding.shape[0])
                cur_hubert_embedding = cur_hubert_embedding[:min_len]
                cur_d2v_embedding = cur_d2v_embedding[:min_len]
            
            # concate along the feature dimension
            merged_embedding = np.concatenate((cur_hubert_embedding, cur_d2v_embedding), axis=-1)
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
            hubert_ckpt=args.hubert_ckpt,
            data2vec_version=args.data2vec_version,
            manifest_path=args.manifest_path,
            kmeans_model_path=args.kmeans_model,
            output_manifest_path=args.output_manifest_path,
            layer_idx=args.layer_idx,
            max_duration=args.max_duration,
        )