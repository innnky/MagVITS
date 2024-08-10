import numpy as np
from pyannote.audio import Model
from pyannote.audio import Inference

import math
import multiprocessing
import os
from random import shuffle
import torch.multiprocessing as mp

import torch
from glob import glob

import torchaudio
from tqdm import tqdm

import utils
import logging

from data_conf import data_root
from module.mel_processing import spectrogram_torch, spec_to_mel_torch
from module.models import SynthesizerTrn

logging.getLogger("numba").setLevel(logging.WARNING)


def process_one(file_path, inference, device):
    spk_emb_path = file_path.replace(".wav", ".spk.npy")
    try:
        np.load(spk_emb_path)
    except:

        embedding = inference(file_path)
        np.save(spk_emb_path, embedding)
    np.save(spk_emb_path, embedding)


def process_batch(filenames):
    print("Loading models ...")
    process_idx = mp.current_process()._identity
    rank = process_idx[0] if len(process_idx) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    print(device)

    model = Model.from_pretrained("pyannote/wespeaker-voxceleb-resnet34-LM")
    model = model.to(device)
    inference = Inference(model, window="whole")

    print("Loaded .")
    with torch.no_grad():
        for filename in tqdm(filenames):
            process_one(filename, inference, device)


in_dir = data_root

if __name__ == "__main__":
    filenames = glob(f"{in_dir}/**/*.wav", recursive=True)  # [:10]
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i: i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk,)) for chunk in chunks
    ]
    for p in processes:
        p.start()

    for p in processes:
        p.join()
