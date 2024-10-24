import torch

from lhotse import load_manifest_lazy, CutSet
from lhotse.cut import MonoCut
from lhotse.features.io import NumpyHdf5Writer
from torch.utils.data import DataLoader
from lhotse.dataset import DynamicBucketingSampler, UnsupervisedWaveformDataset

import fairseq

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
    print(rep.shape)
    
@torch.no_grad()
def collect_results(manifest_path, embedding_path, layer_idx=21, max_duration=200):
    # load the pre-trained checkpoints
    ckpt_path = "hubert_base_ls960.pt"
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
    with NumpyHdf5Writer(embedding_path) as writer:
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
                new_cut = MonoCut(
                    id=cut.id,
                    start=cut.start,
                    duration=cut.duration,
                    channel=cut.channel,
                )
                new_cut.hubert_embedding = writer.store_array(
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
    new_cuts.to_jsonl(f"dev-clean-hubert-base-layer-{layer_idx}.jsonl.gz")
            
        
if __name__=="__main__":
    manifest_path = "data/fbank/librispeech_cuts_dev-clean.jsonl.gz"
    embedding_path = "hubert_embeddings/hubert-base-dev-clean.h5"
    max_duration = 100
    
    collect_results(
        manifest_path=manifest_path,
        embedding_path=embedding_path,
        layer_idx=-1,
        max_duration=max_duration,
    )