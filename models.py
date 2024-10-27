import torch
import torch.nn.functional as F

from whisper.audio import log_mel_spectrogram, pad_or_trim, N_FRAMES

from icefall import make_pad_mask

class Teacher(torch.nn.Module):
    def __init__(
        self,
        model: torch.nn.Module
    ):
        super().__init__()
        self.model = model
        
    def get_embeddings(self):
        raise NotImplementedError()

class WhisperTeacher(Teacher):
    def __init__(
        self,
        model: torch.nn.Module,
        n_mels: int = 80,
    ):
        super().__init__(model)
        self.n_mels = n_mels
        
    def forward_encoder(
        self,
        x: torch.Tensor,
        x_lens: torch.Tensor,
        layer_idx: int = -1,
    ):
        """
        x : torch.Tensor, shape = (batch_size, n_mels, n_ctx)
            the mel spectrogram of the audio
        x_lens: torch.Tensor, shape = (batch_size)
        layer_idx: which layer's feature to extract
        """
        x = F.gelu(self.model.conv1(x))
        x = F.gelu(self.model.conv2(x))
        x = x.permute(0, 2, 1)
        x_lens = torch.floor((x_lens + 1)/2).int()
        
        # make the model compatible with any input length
        mask = make_pad_mask(x_lens, max_len=1500).to(x.device)
        pos_emb = self.model.positional_embedding.masked_fill(mask.unsqueeze(-1), 0.0)
        x = (x + pos_emb[:,:x_lens.max(),:]).to(x.dtype)
        
        results = []
        for block in self.model.blocks:
            x = block(x)
            results.append(x)
        if layer_idx == -1 or layer_idx == len(results) - 1: # use the last layer
            x = self.model.ln_post(x)
        else:
            x = results[layer_idx] # zero-based index

        return x, x_lens
            
    @torch.no_grad()
    def get_embeddings(
        self,
        audio: torch.Tensor,
        audio_lens: torch.Tensor,
        layer_idx: int = -1
    ):
        # return the embeddings of the input audio
        # audio_lens is the number of raw samples (i.e waveform)
        device = next(self.model.parameters()).device
        audio = audio.to(device)
        audio_lens = audio_lens.to(device)
        mel = log_mel_spectrogram(audio, n_mels=self.n_mels) # (N, n_mel, T)
        
        if mel.ndim == 2:
            mel = mel.unsqueeze(0)
        
        mel_lens = torch.floor(audio_lens/160).int()
        assert mel_lens.max() <= mel.size(-1)

        features, feature_lens = self.forward_encoder(
            mel,
            mel_lens,
            layer_idx=layer_idx,
        )
        
        return features, feature_lens