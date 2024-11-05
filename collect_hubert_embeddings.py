import argparse
import logging

import torch

from lhotse import load_manifest_lazy, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer, LilcomChunkyWriter
from lhotse.utils import fastcopy
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

import fairseq

from utils import make_pad_mask

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--hubert-version",
        type=str,
        choices=["large", "base", "large-ft960"],
        required=True,
    )
    
    parser.add_argument(
        "--hubert-ckpt",
        type=str,
        help="path to the hubert checkpoint",
        required=True,
    )
    
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=-1,
        help="The index starts from 1, so if you want the 12-th layer feature, just set it to 12"
    )
    
    parser.add_argument(
        "--subset",
        type=str,
        required=True,
    )
    
    return parser.parse_args()

def test_hubert():
    # load the pre-trained checkpoints
    ckpt_path = "hubert_base_ls960.pt"
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    model.eval()

    # extract the representation of last layer
    wav_input_16khz = torch.randn(1,10000)
    rep, _ = model.extract_features(wav_input_16khz)
    
    # extract the representation of each layer
    wav_input_16khz = torch.randn(1,10000)
    padding_mask = torch.zeros(1, 10000).bool()
    layer_idx=12
    rep, padding_mask = model.extract_features(wav_input_16khz, padding_mask=padding_mask, output_layer=layer_idx)
    logging.info(rep.shape)
    
@torch.no_grad()
def collect_results(
    ckpt_path,
    manifest_path,
    embedding_path,
    output_manifest_path,
    layer_idx=21,
    max_duration=200
):
    # load the pre-trained checkpoints
    models, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
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
    with LilcomChunkyWriter(embedding_path) as writer:
        for i, batch in enumerate(dl):
            cuts = batch["cuts"]
            audio_input_16khz = batch["audio"].to(device)
            audio_lens = batch["audio_lens"].to(device)
            padding_mask = make_pad_mask(audio_lens)
            
            layer_results, padding_mask = model.extract_features(
                audio_input_16khz,
                padding_mask=padding_mask,
                output_layer=layer_idx,
            )
            layer_results = layer_results.cpu().numpy()
            
            embedding_lens = (~padding_mask).sum(dim=-1)
            
            for j, cut in enumerate(cuts):
                hubert_embedding = writer.store_array(
                    key=cut.id,
                    value=layer_results[j, :embedding_lens[j]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=0,
                )
                new_cut = fastcopy(
                    cut,
                    custom={"hubert_embedding": hubert_embedding}
                )
                new_cuts.append(new_cut)
                num_cuts += 1
                if num_cuts and num_cuts % 100 == 0:
                    logging.info(f"Cuts processed until now: {num_cuts}")
    
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest_path)
    logging.info(f"Manifest saved to {output_manifest_path}")
            
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    hubert_version = args.hubert_version
    ckpt_path = args.hubert_ckpt
    layer_idx = args.layer_idx
    subset = args.subset
    
    manifest_path = f"data/fbank/librispeech_cuts_{subset}.jsonl.gz"
    embedding_path = f"embeddings/hubert_embeddings/hubert-{hubert_version}-layer-{layer_idx}-{subset}.h5"
    output_manifest_path = f"manifests/{subset}-hubert-{hubert_version}-layer-{layer_idx}.jsonl.gz"

    max_duration = 100
    
    collect_results(
        ckpt_path=ckpt_path,
        manifest_path=manifest_path,
        embedding_path=embedding_path,
        output_manifest_path=output_manifest_path,
        layer_idx=layer_idx,
        max_duration=max_duration,
    )