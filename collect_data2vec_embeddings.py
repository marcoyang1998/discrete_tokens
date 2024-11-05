import argparse
import logging

import torch
import numpy as np

from lhotse import load_manifest_lazy, CutSet
from lhotse.utils import fastcopy
from lhotse.cut import MonoCut
from lhotse.features.io import LilcomChunkyWriter
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from transformers import AutoProcessor, Data2VecAudioModel

def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--data2vec-version",
        type=str,
        choices=["large", "base"],
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

def test_data2vec():
    # load the pre-trained checkpoints
    import pdb; pdb.set_trace()
    processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
    model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")
    model.eval()

    # extract the representation of last layer
    wav_input_16khz = [np.random.randn(10000), np.random.randn(12000)]
    
    inputs = processor(wav_input_16khz, sampling_rate=16000, padding=True, return_attention_mask=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(output_hidden_states=True, **inputs)
    print(len(outputs.hidden_states)) # should be num_layers+1, because the input to encoder is also returned
    print(outputs.last_hidden_state.shape)
    assert torch.all(outputs.hidden_states[-1] == outputs.last_hidden_state)
    
@torch.no_grad()
def collect_results(
    model_version,
    manifest_path,
    embedding_path,
    output_manifest_path,
    layer_idx=21,
    max_duration=200  
):
    # load the pre-trained checkpoints
    processor = AutoProcessor.from_pretrained(f"facebook/data2vec-audio-{model_version}")
    model = Data2VecAudioModel.from_pretrained(f"facebook/data2vec-audio-{model_version}")
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
            audios = []
            for cut in cuts:
                audio = cut.load_audio()
                audios.append(audio.reshape(-1))
            inputs = processor(
                audios, 
                sampling_rate=16000,
                padding=True,
                return_attention_mask=True,
                return_tensors="pt"
            ).to(device)
            
            outputs = model(
                output_hidden_states=True,
                **inputs,
            )
            all_layer_results = outputs.hidden_states
            # If the model has 12 layers, it will return 13 features
            # with the first one as the input features after conv modules
            # So if you want to extract the 12-th layer's representation
            # Use layer_idx=12
            layer_results = all_layer_results[layer_idx].cpu().numpy()
            
            # get the padding mask after the conv modules
            padding_mask = model._get_feature_vector_attention_mask(layer_results.shape[1], inputs["attention_mask"])
            embedding_lens = padding_mask.sum(dim=-1)
            
            for j, cut in enumerate(cuts):
                data2vec_embedding = writer.store_array(
                    key=cut.id,
                    value=layer_results[j, :embedding_lens[j]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=0,
                )
                new_cut = fastcopy(
                    cut,
                    custom={"data2vec_embedding": data2vec_embedding}
                )
                new_cuts.append(new_cut)
                num_cuts += 1
                if num_cuts and num_cuts % 100 == 0:
                    print(f"Cuts processed until now: {num_cuts}")
    
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(output_manifest_path)
    logging.info(f"Manifest saved to {output_manifest_path}")
            
        
if __name__=="__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    logging.basicConfig(format=formatter, level=logging.INFO)
    
    args = get_parser()
    data2vec_version = args.data2vec_version
    layer_idx = args.layer_idx
    subset = args.subset
    
    manifest_path = f"data/fbank/librispeech_cuts_{subset}.jsonl.gz"
    embedding_path = f"embeddings/data2vec_embeddings/data2vec-{data2vec_version}-layer-{layer_idx}-{subset}.h5"
    output_manifest_path = f"manifests/{subset}-data2vec-{data2vec_version}-layer-{layer_idx}.jsonl.gz"
    max_duration = 100
    
    collect_results(
        model_version=data2vec_version,
        manifest_path=manifest_path,
        embedding_path=embedding_path,
        layer_idx=layer_idx,
        max_duration=max_duration,
        output_manifest_path=output_manifest_path,
    )