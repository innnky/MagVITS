from text import cleaned_text_to_sequence
from text.cleaner import text_to_sequence, clean_text
from gen_phonemes import get_bert_feature
import torch.nn.functional as F
from module import commons
import torch    
import utils
from module.models import SynthesizerTrn
from module.mel_processing import spectrogram_torch, spec_to_mel_torch
import soundfile
import torchaudio
from pyannote.audio import Model
from pyannote.audio import Inference
import numpy

def text2phoneid(text, lang='zh'):
    phones, word2ph, norm_text = clean_text(text, lang)
    print(phones)

    bert = get_bert_feature(norm_text, word2ph, 'cpu', lang)
    phonemes = cleaned_text_to_sequence(phones)
    phonemes = commons.intersperse(phonemes, 0)
    bert = F.interpolate(bert.unsqueeze(0), scale_factor=2, mode='nearest')
    bert = F.pad(bert, (0, 1), value=0).squeeze(0)
    return phones, phonemes, bert



def load_model(device="cuda", config_path="configs/s2.json", model_path=None):
    device = torch.device(device)
    print('loading models...')
    hps = utils.get_hparams_from_file(config_path)
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    if model_path is None:
        model_path = utils.latest_checkpoint_path(hps.s2_ckpt_dir, "G_*.pth")
    utils.load_checkpoint(model_path, net_g,
                          None, False)
    net_g.eval()
    spk_emb_model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    spk_emb_model = spk_emb_model.to(device)
    inference = Inference(spk_emb_model, window="whole")

    return hps, net_g, inference



def get_spepc(hps, filename):
    audio, sampling_rate = utils.load_wav_to_torch(filename)
    audio = audio.unsqueeze(0)
    if sampling_rate != hps.data.sampling_rate:
        audio = torchaudio.functional.resample(audio, sampling_rate, hps.data.sampling_rate)
    audio_norm = audio
    spec = spectrogram_torch(audio_norm, hps.data.filter_length,
                             hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                             center=False)
    return spec



@torch.no_grad()
@torch.inference_mode()
def decode_to_file(codes, ref_path, save_path):
    device = codes.device
    hps, net_g, ssl = load_model(device=device)
    ref = get_spepc(hps, ref_path).to(device)

    audio = net_g.decode_codes(codes, ref).detach().cpu().numpy()[0, 0]
    soundfile.write(save_path, audio, hps.data.sampling_rate)

import os
if __name__ == '__main__':

    device = 'cpu'
    outdir = 'out'
    os.makedirs(outdir, exist_ok=True)
    txt_list = [
         "先帝创业未半而中道崩殂，今天下三分，益州疲弊，此诚危急存亡之秋也。然侍卫之臣不懈于内，忠志之士忘身于外者，盖追先帝之殊遇，欲报之于陛下也。",
        '话说天下大势，分久必合，合久必分。周末七国分争，并入于秦。及秦灭之后，楚汉分争，又并入于汉。汉朝自高祖斩白蛇而起义，一统天下，后来光武中兴，传至献帝，遂分为三国。',
    ]

    prompt_list = [ "dataset_raw/zh/Azusa/Azusa_113.wav",
                   'dataset_raw/zh/Azusa/Azusa_288.wav',]
    


    hps, model,spk_emb_model = load_model(device=device)

    for name, text in enumerate(txt_list):
        for i, prompt_wav_path in enumerate(prompt_list):
            out_path = f'{outdir}/{name}_{i}.wav'
            phlist, phones, bert = text2phoneid(text)
            print(len(phones))
            ref = get_spepc(hps, prompt_wav_path).to(device)
            spk_emb = spk_emb_model(prompt_wav_path)
            spk_emb = torch.FloatTensor(spk_emb).to(device).unsqueeze(0)

            all_phoneme_ids = torch.LongTensor(phones).to(device).unsqueeze(0)
            bert = bert.to(device).unsqueeze(0)
            x_lengths = torch.LongTensor([all_phoneme_ids.shape[-1]]).to(device)

            with torch.no_grad():
                wavs = model.infer(all_phoneme_ids, x_lengths,  ref, bert,spk_emb,noise_scale=.4)
                soundfile.write(out_path, wavs[0,0].cpu().numpy(), hps.data.sampling_rate)