import time

import librosa
import torch
import torch.nn.functional as F
import soundfile as sf
import logging

logging.getLogger("numba").setLevel(logging.WARNING)

from transformers import (
    Wav2Vec2FeatureExtractor,
    HubertModel,
)
from torchaudio.transforms import Resample

import utils
import torch.nn as nn

class CNHubert(nn.Module):
    def __init__(self, input_sample_rate=32000):
        super().__init__()
        self.model = HubertModel.from_pretrained("TencentGameMate/chinese-hubert-base")
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("TencentGameMate/chinese-hubert-base")
        if input_sample_rate != 16000:
            self.resample = Resample(orig_freq=input_sample_rate, new_freq=16000)
        else:
            self.resample = None
        self.input_sample_rate = input_sample_rate
        self.log_flag = False
    def forward(self, x):
        if self.resample is not None:
            if not self.log_flag:
                print(f"Resampling from {x.shape[-1]} to 16000")
                self.log_flag = True
            x = self.resample(x)
        input_values = self.feature_extractor(x, return_tensors="pt", sampling_rate=16000).input_values.to(x.device).to(x.dtype)
        feats = self.model(input_values)["last_hidden_state"]
        return feats



def get_model(input_sample_rate=32000):
    model = CNHubert(input_sample_rate=input_sample_rate)
    model.eval()
    return model

def get_content(hmodel, src_path):
    input_sample_rate = hmodel.input_sample_rate
    wav, _ = librosa.load(src_path, sr=input_sample_rate)
    device = hmodel.parameters().__next__().device
    dtype = hmodel.parameters().__next__().dtype
    wav_16k_tensor = torch.from_numpy(wav).to(device).to(dtype)
    with torch.no_grad():
        feats = hmodel(wav_16k_tensor)
    return feats.transpose(1,2)


if __name__ == '__main__':
    model = get_model()
    src_path = "/Users/Shared/原音频2.wav"
    wav_16k_tensor = utils.load_wav_to_torch_and_resample(src_path, 16000)
    model = model
    wav_16k_tensor = wav_16k_tensor
    feats = get_content(model,wav_16k_tensor)
    print(feats.shape)

