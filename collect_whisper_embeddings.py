import argparse
import logging

import torch
import whisper

from lhotse import load_manifest, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from models import WhisperTeacher

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--whisper-version",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
    )
    
    return parser.parse_args()

def remove_short_and_long_utt(c):
    if c.duration < 1.0 or c.duration > 30.0:
        return False
    return True

def collect_whisper_embeddings(whisper_version, manifest_path, embedding_path, output_manifest_path, layer_idx=21, max_duration=200):
    # load the pre-trained checkpoints
    
    device = torch.device("cuda")
    
    whisper_model = whisper.load_model(whisper_version, device)
    n_mels = whisper_model.dims.n_mels
    whisper_model = whisper_model.encoder
    whisper_model.eval()
        
    logging.info(f"Number of whisper encoder params: {sum(p.numel() for p in whisper_model.parameters())}")
    logging.info(f"Successfully loaded Whisper {whisper_version} model.")
    
    model = WhisperTeacher(model=whisper_model, n_mels=n_mels)
    
    manifest = load_manifest(manifest_path)
    manifest = manifest.filter(remove_short_and_long_utt)
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
    
    new_cuts = []
    num_cuts = 0
    with NumpyHdf5Writer(embedding_path) as writer:
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio = batch["audio"]
            audio_lens = batch["audio_lens"]
            
            embeddings, embedding_lens = model.get_embeddings(
                audio, 
                audio_lens,
                layer_idx=layer_idx # which layer's embedding to be stored, index starts from zero
            )
            embeddings = embeddings.detach().to("cpu").numpy()
            
            for j, cut in enumerate(cuts):
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.whisper_embedding = writer.store_array(
                    key=cut.id,
                    value=embeddings[j][: embedding_lens[j]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=cut.start,
                )
                new_cuts.append(new_cut)
                num_cuts += 1
                if num_cuts and num_cuts % 200 == 0:
                    logging.info(f"Cuts processed until now: {num_cuts}")
    
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest_path)
    logging.info(f"Manifest saved to {output_manifest_path}")
            
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    whisper_version = args.whisper_version
    layer_idx = args.layer_idx
    manifest_path = "data/fbank/librispeech_cuts_dev-clean.jsonl.gz"
    embedding_path = f"whisper_embeddings/whisper-{whisper_version}-dev-clean.h5"
    output_manifest_path = f"manifests/dev-clean-whisper-{whisper_version}-layer-{layer_idx}.jsonl.gz"
    
    max_duration = 100
    collect_whisper_embeddings(
        whisper_version=whisper_version,
        manifest_path=manifest_path,
        embedding_path=embedding_path,
        output_manifest_path=output_manifest_path,
        layer_idx=layer_idx,
        max_duration=max_duration,
    )