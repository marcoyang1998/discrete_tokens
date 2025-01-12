import argparse
import logging

import torch
import os

from lhotse import load_manifest, CutSet
from lhotse.features.io import LilcomChunkyWriter, NumpyHdf5Writer
from lhotse.utils import fastcopy
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from models import get_model
from utils import remove_long_utterances

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        choices=["hubert", "wavlm", "w2v-bert", "data2vec", "whisper"]
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
        "--subset",
        type=str,
        required=True,
    )
    
    parser.add_argument(
        "--input-manifest",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-duration",
        type=int,
        default=200,
    )
    
    return parser.parse_args()

def test_hubert():
    # load the pre-trained checkpoints
    import fairseq
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
    model_name,
    model_version,
    manifest_path,
    embedding_path,
    output_manifest_path,
    layer_idx=21,
    max_duration=200
):
    # load the pre-trained checkpoints
    model = get_model(args)
    model.eval()
    
    model.eval()
    
    manifest = load_manifest(manifest_path)
    manifest = manifest.filter(remove_long_utterances)
    
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
            layer_results, embedding_lens = model.extract_features(batch, layer_idx)
            
            for j, cut in enumerate(cuts):
                embedding = writer.store_array(
                    key=cut.id,
                    value=layer_results[j, :embedding_lens[j]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=0,
                )
                new_cut = fastcopy(
                    cut,
                    custom={f"embedding": embedding} # remove model name to be more generic
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
    logging.info(vars(args))
    model_name = args.model_name
    model_version = args.model_version
    layer_idx = args.layer_idx
    subset = args.subset
    
    input_manifest = args.input_manifest
    embedding_path = f"embeddings/{model_name}_embeddings/{model_name}-{model_version}-layer-{layer_idx}-{subset}"
    output_manifet = f"manifests/{subset}-{model_name}-{model_version}-layer-{layer_idx}.jsonl.gz"

    if os.path.exists(output_manifet):
        logging.info(f"Manifest already exists at: {output_manifet}")
        logging.info("Skip collecting the embeddings")
        
    else:
        collect_results(
            model_name=model_name,
            model_version=model_version,
            manifest_path=input_manifest,
            embedding_path=embedding_path,
            output_manifest_path=output_manifet,
            layer_idx=layer_idx,
            max_duration=args.max_duration,
        )