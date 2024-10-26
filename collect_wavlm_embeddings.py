import logging

import torch
from WavLM import WavLM, WavLMConfig

from lhotse import load_manifest_lazy, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from icefall.utils import make_pad_mask

def test_wavlm():
    # load the pre-trained checkpoints
    checkpoint = torch.load('WavLM-Base+.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    device = torch.device("cuda")
    model.to(device)

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1,10000).to(device)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
    rep = model.extract_features(wav_input_16khz)[0]

    # extract the representation of each layer
    wav_input_16khz = torch.randn(1,10000)
    padding_mask = torch.zeros(1, 10000).bool()
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
    rep, layer_results = model.extract_features(wav_input_16khz, padding_mask=padding_mask, output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
    layer_reps = [x.transpose(0, 1) for x, _ in layer_results]
    
@torch.no_grad()
def collect_results(model_name, manifest_path, embedding_path, output_manifest_path, layer_idx=21, max_duration=200):
    # load the pre-trained checkpoints
    if model_name == "wavlm-base-plus":
        checkpoint = torch.load('WavLM-Base+.pt')
    elif model_name == "wavlm-base":
        checkpoint = torch.load('WavLM-Base.pt')
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
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
    
    new_cuts = []
    num_cuts = 0
    with NumpyHdf5Writer(embedding_path) as writer:
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio_input_16khz = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            padding_mask = make_pad_mask(audio_lens)
            
            if cfg.normalize:
                audio_input_16khz = torch.nn.functional.layer_norm(audio_input_16khz, audio_input_16khz.shape)
            
            (rep, layer_results), padding_mask = model.extract_features(
                audio_input_16khz,
                padding_mask=padding_mask,
                output_layer=model.cfg.encoder_layers,
                ret_layer_results=True
            )
            
            layer_results = [res.permute(1,0,2).cpu().numpy() for res, _ in layer_results] # list of (B,T,C)
            layer_results = layer_results[layer_idx] # (B,T,C)
            embedding_lens = (~padding_mask).sum(dim=-1)
            
            for j, cut in enumerate(cuts):
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.wavlm_embedding = writer.store_array(
                    key=cut.id,
                    value=layer_results[j][:embedding_lens[j]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=0,
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
    
    model_name = "wavlm-base"
    manifest_path = "data/fbank/librispeech_cuts_dev-clean.jsonl.gz"
    embedding_path = f"wavlm_embeddings/{model_name}-dev-clean.h5"
    layer_idx = -1
    output_manifest_path = f"manifests/dev-clean-{model_name}-layer-{layer_idx}.jsonl.gz"
    
    max_duration = 100
    collect_results(
        model_name=model_name,
        manifest_path=manifest_path,
        embedding_path=embedding_path,
        output_manifest_path=output_manifest_path,
        layer_idx=layer_idx,
        max_duration=max_duration,
    )