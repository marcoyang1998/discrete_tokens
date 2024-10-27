import torch
import numpy as np

from lhotse import load_manifest_lazy, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

from transformers import AutoProcessor, Data2VecAudioModel

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """
    Args:
      lengths:
        A 1-D tensor containing sentence lengths.
      max_len:
        The length of masks.
    Returns:
      Return a 2-D bool tensor, where masked positions
      are filled with `True` and non-masked positions are
      filled with `False`.

    >>> lengths = torch.tensor([1, 3, 2, 5])
    >>> make_pad_mask(lengths)
    tensor([[False,  True,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False, False, False]])
    """
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)


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
def collect_results(manifest_path, embedding_path, layer_idx=21, max_duration=200):
    # load the pre-trained checkpoints
    processor = AutoProcessor.from_pretrained("facebook/data2vec-audio-base-960h")
    model = Data2VecAudioModel.from_pretrained("facebook/data2vec-audio-base-960h")
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
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.data2vec_embedding = writer.store_array(
                    key=cut.id,
                    value=layer_results[j, :embedding_lens[j]],
                    temporal_dim=0,
                    frame_shift=0.02,
                    start=0,
                )
                new_cuts.append(new_cut)
                num_cuts += 1
                if num_cuts and num_cuts % 100 == 0:
                    print(f"Cuts processed until now: {num_cuts}")
    
    new_cuts = CutSet.from_cuts(new_cuts)
    new_cuts.to_jsonl(f"manifests/dev-clean-data2vec-base-layer-{layer_idx}.jsonl.gz")
            
        
if __name__=="__main__":
    manifest_path = "data/fbank/librispeech_cuts_dev-clean.jsonl.gz"
    embedding_path = "data2vec_embeddings/data2vec-base-dev-clean.h5"
    max_duration = 100
    
    # test_data2vec()
    collect_results(
        manifest_path=manifest_path,
        embedding_path=embedding_path,
        layer_idx=12,
        max_duration=max_duration,
    )