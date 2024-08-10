import math
import multiprocessing
import argparse
from random import shuffle
import torch.multiprocessing as mp

import torch
from glob import glob
from tqdm import tqdm

import utils
from data_conf import data_root
from module.cnhubert import get_model, get_content
import logging

logging.getLogger("numba").setLevel(logging.WARNING)
import librosa


def process_one(file_path, model):

    # file_path16k = file_path.replace('genshin_data', 'genshin_data16k')

    ssl_path = file_path.replace(".wav", ".ssl.pt")
    ssl_content = get_content(model, file_path)
    assert not torch.isnan(ssl_content).any(), f"NaN in {file_path}"
    torch.save(ssl_content.half().cpu(), ssl_path)

def process_batch(filenames):
    print("Loading content model...")
    rank = mp.current_process()._identity
    rank = rank[0] if len(rank) > 0 else 0
    gpu_id = rank % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    ssl_model = get_model()
    ssl_model = ssl_model.to(device)
    ssl_model.eval()
    print("Loaded content model.")
    for filename in tqdm(filenames):
        try:
            process_one(filename, ssl_model)
        except Exception as e:
            print(f"Error processing {filename}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config", type=str, default="configs/s2.json", help="path to config"
    )
    args = parser.parse_args()
    filenames = glob(f"{data_root}/**/*.wav", recursive=True)  # [:10]
    hps = utils.get_hparams_from_file(args.config)
    shuffle(filenames)
    multiprocessing.set_start_method("spawn", force=True)

    num_processes = 1
    chunk_size = int(math.ceil(len(filenames) / num_processes))
    chunks = [
        filenames[i : i + chunk_size] for i in range(0, len(filenames), chunk_size)
    ]
    print([len(c) for c in chunks])
    processes = [
        multiprocessing.Process(target=process_batch, args=(chunk, )) for chunk in chunks
    ]
    for p in processes:
        p.start()