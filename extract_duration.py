import yaml
import librosa
from text import cleaned_text_to_sequence
from asr.models import build_model
import torch
import torchaudio
from tqdm import tqdm
from monotonic_align import mask_from_lens
from monotonic_align.core import maximum_path_c
import torch.nn.functional as F
import numpy as np
import os

from asr.trainer import calc_wer


def maximum_path(neg_cent, mask):
    """ Cython optimized version.
    neg_cent: [b, t_t, t_s]
    mask: [b, t_t, t_s]
    """
    device = neg_cent.device
    dtype = neg_cent.dtype
    neg_cent = np.ascontiguousarray(neg_cent.data.cpu().numpy().astype(np.float32))
    path = np.ascontiguousarray(np.zeros(neg_cent.shape, dtype=np.int32))

    t_t_max = np.ascontiguousarray(mask.sum(1)[:, 0].data.cpu().numpy().astype(np.int32))
    t_s_max = np.ascontiguousarray(mask.sum(2)[:, 0].data.cpu().numpy().astype(np.int32))
    maximum_path_c(path, neg_cent, t_t_max, t_s_max)
    return torch.from_numpy(path).to(device=device, dtype=dtype)


def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result


def calc_mono_loss(s2s_attn, input_lengths, mel_input_length, text_mask, mel_mask, n_down):
    s2s_attn = s2s_attn.transpose(-1, -2)
    s2s_attn = s2s_attn[..., 1:]
    s2s_attn = s2s_attn.transpose(-1, -2)

    with torch.no_grad():
        attn_mask = (~mel_mask).unsqueeze(-1).expand(mel_mask.shape[0], mel_mask.shape[1],
                                                     text_mask.shape[-1]).float().transpose(-1, -2)
        attn_mask = attn_mask.float() * (~text_mask).unsqueeze(-1).expand(text_mask.shape[0],
                                                                          text_mask.shape[1],
                                                                          mel_mask.shape[-1]).float()
        attn_mask = (attn_mask < 1)

    s2s_attn.masked_fill_(attn_mask, 0.0)

    with torch.no_grad():
        mask_ST = mask_from_lens(s2s_attn, input_lengths, mel_input_length)
        s2s_attn_mono = maximum_path(s2s_attn, mask_ST)
    loss_mono = F.l1_loss(s2s_attn, s2s_attn_mono) * 10

    return loss_mono, s2s_attn_mono


def get_attention_mono(model, text_input, text_input_length, mel_input, mel_input_length):
    mel_input_length = mel_input_length // (2 ** model.n_down)
    future_mask = model.get_future_mask(
        mel_input.size(2) // (2 ** model.n_down), unmask_future_steps=0).to(text_input.device)
    mel_mask = model.length_to_mask(mel_input_length)
    text_mask = model.length_to_mask(text_input_length)
    ppgs, s2s_pred, s2s_attn = model(
        mel_input, src_key_padding_mask=mel_mask, text_input=text_input)
    loss_mono, s2s_attn_mono = calc_mono_loss(s2s_attn, text_input_length, mel_input_length, text_mask, mel_mask,
                                              model.n_down)
    _, amax_ppgs = torch.max(ppgs, dim=2)
    wers = [calc_wer(target[:text_length],
                     pred[:mel_length],
                     ignore_indexes=list(range(5))) \
            for target, pred, text_length, mel_length in zip(
            text_input.cpu(), amax_ppgs.cpu(), text_input_length.cpu(), mel_input_length.cpu())]
    m_wer = np.mean(wers)
    return s2s_attn_mono, m_wer


to_mel = torchaudio.transforms.MelSpectrogram(
    n_mels=128, n_fft=2048, win_length=2048, hop_length=640)
mean, std = -4, 4


def preprocess(wave):
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


config_path = 'configs/asr.yml'
ckpt_path = 'logs/asr/last.ckpt'
dump_dir = 'dump'
phoneme_path = f'{dump_dir}/phoneme.npy'
train_path = f'{dump_dir}/s2_train_files.list'
val_path = f'{dump_dir}/s2_val_files.list'
config = yaml.safe_load(open(config_path))
model = build_model(model_params=config['model_params'] or {})
state_dict = torch.load(ckpt_path, map_location="cpu")['state_dict']
state_dict = {k.replace('model.', ''):v for k, v in state_dict.items()}
model.load_state_dict(state_dict)
device = 'cuda:0'
model = model.to(device)
model.eval()
phoneme_data = np.load(phoneme_path, allow_pickle=True).item()

all_files = [line.strip() for line in open(train_path)]
import random

random.shuffle(all_files)
processed_cnt = 0
all_files = [line.strip() for line in open(val_path)] + all_files
with torch.no_grad():
    for line in tqdm(all_files):
        wave_path = line.strip()
        try:
            phonemes = phoneme_data[wave_path]
        except:
            print('phoneme not exist ,skip:', wave_path)
            continue
        if not os.path.exists(wave_path):
            print('skip:', wave_path)
            continue
        wave, sr = librosa.load(wave_path, sr=None)
        if wave.shape[-1] == 2:
            wave = wave[:, 0].squeeze()
        assert sr == 32000
        phoneme = phonemes.split(' ')
        phoneme_ids = cleaned_text_to_sequence(phoneme)
        phoneme_ids = intersperse(phoneme_ids, 0)

        text = torch.LongTensor(phoneme_ids)
        mel_tensor = preprocess(wave).squeeze()

        ph_len = len(phoneme_ids)
        mel_len = mel_tensor.shape[-1]
        ps = mel_len/ph_len

        if ps < 1.2 or ps>10:
            print(ph_len, mel_len, )
            print('skip:',wave_path)
            continue
        acoustic_feature = mel_tensor.squeeze()
        length_feature = acoustic_feature.size(1)
        # acoustic_feature = acoustic_feature[:, :(length_feature - length_feature % 2)]

        # print(acoustic_feature.size(), text.size())

        text_input = text.unsqueeze(0).to(device)
        text_input_length = torch.LongTensor([len(phoneme_ids)]).to(device)
        mel_input = mel_tensor.unsqueeze(0).to(device)
        mel_input_length = torch.LongTensor([mel_input.size(2)]).to(device)

        s2s_attn_mono, m_wer = get_attention_mono(model, text_input, text_input_length, mel_input, mel_input_length)
        duration = s2s_attn_mono[0].long().sum(-1).detach().cpu().numpy().tolist()
        # duration = s2s_attn_mono[0].long().sum(-1).detach().cpu().numpy().tolist()
        # print(duration, len(duration), sum(duration))
        duration = s2s_attn_mono[0].long().sum(-1).detach().cpu()

        save_path = wave_path.replace('.wav', '.dur.pt')
        torch.save(duration, save_path)
        processed_cnt += 1
        # text_input, text_input_length, mel_input, mel_input_length
        # print(s2s_attn_mono.shape, duration, len(duration), sum(duration), len(phoneme), acoustic_feature.shape)
        # break

print(processed_cnt)